import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from seaborn import color_palette
import cv2
import os
import mlflow
import mlflow.tensorflow
from datetime import datetime

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Handle TensorFlow 2.x compatibility
if tf.__version__.startswith('2'):
    tf = tf.compat.v1
    tf.disable_eager_execution()
    # Define conversion functions for TF 2.x
    def tf_to_float(x):
        return tf.cast(x, tf.float32)
else:
    # Define conversion functions for TF 1.x
    def tf_to_float(x):
        return tf.to_float(x)

# Constants
_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1
_ANCHORS = [(10, 13), (16, 30), (33, 23),
            (30, 61), (62, 45), (59, 119),
            (116, 90), (156, 198), (373, 326)]
_MODEL_SIZE = (416, 416)

def batch_norm(inputs, training, data_format):
    """Fixed batch normalization"""
    return tf.keras.layers.BatchNormalization(
        axis=1 if data_format == 'channels_first' else 3,
        momentum=_BATCH_NORM_DECAY,
        epsilon=_BATCH_NORM_EPSILON,
        scale=True)(inputs, training=training)

def fixed_padding(inputs, kernel_size, data_format):
    """Fixed padding implementation"""
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                      [pad_beg, pad_end],
                                      [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                      [pad_beg, pad_end], [0, 0]])
    return padded_inputs

def conv2d_fixed_padding(inputs, filters, kernel_size, data_format, strides=1):
    """Fixed conv2d implementation"""
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)
    
    return tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same' if strides == 1 else 'valid',
        use_bias=False,
        data_format=data_format)(inputs)

def darknet53_residual_block(inputs, filters, training, data_format, strides=1):
    """Creates a residual block for Darknet."""
    shortcut = inputs

    inputs = conv2d_fixed_padding(
        inputs, filters=filters, kernel_size=1, strides=strides,
        data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs = conv2d_fixed_padding(
        inputs, filters=2 * filters, kernel_size=3, strides=strides,
        data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs += shortcut

    return inputs

def darknet53(inputs, training, data_format):
    """Creates Darknet53 model for feature extraction."""
    inputs = conv2d_fixed_padding(inputs, filters=32, kernel_size=3,
                                  data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)
    inputs = conv2d_fixed_padding(inputs, filters=64, kernel_size=3,
                                  strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs = darknet53_residual_block(inputs, filters=32, training=training,
                                      data_format=data_format)

    inputs = conv2d_fixed_padding(inputs, filters=128, kernel_size=3,
                                  strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    for _ in range(2):
        inputs = darknet53_residual_block(inputs, filters=64,
                                          training=training,
                                          data_format=data_format)
        
    inputs = conv2d_fixed_padding(inputs, filters=256, kernel_size=3,
                                  strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    for _ in range(8):
        inputs = darknet53_residual_block(inputs, filters=128,
                                          training=training,
                                          data_format=data_format)

    route1 = inputs

    inputs = conv2d_fixed_padding(inputs, filters=512, kernel_size=3,
                                  strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    for _ in range(8):
        inputs = darknet53_residual_block(inputs, filters=256,
                                          training=training,
                                          data_format=data_format)

    route2 = inputs

    inputs = conv2d_fixed_padding(inputs, filters=1024, kernel_size=3,
                                  strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    for _ in range(4):
        inputs = darknet53_residual_block(inputs, filters=512,
                                          training=training,
                                          data_format=data_format)

    return route1, route2, inputs

def yolo_convolution_block(inputs, filters, training, data_format):
    """Creates convolution operations layer used after Darknet."""
    inputs = conv2d_fixed_padding(inputs, filters=filters, kernel_size=1,
                                  data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs = conv2d_fixed_padding(inputs, filters=2 * filters, kernel_size=3,
                                  data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs = conv2d_fixed_padding(inputs, filters=filters, kernel_size=1,
                                  data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs = conv2d_fixed_padding(inputs, filters=2 * filters, kernel_size=3,
                                  data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs = conv2d_fixed_padding(inputs, filters=filters, kernel_size=1,
                                  data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    route = inputs

    inputs = conv2d_fixed_padding(inputs, filters=2 * filters, kernel_size=3,
                                  data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    return route, inputs

def yolo_layer(inputs, n_classes, anchors, img_size, data_format):
    """Fixed yolo_layer implementation"""
    n_anchors = len(anchors)
    
    inputs = tf.keras.layers.Conv2D(
        filters=n_anchors * (5 + n_classes),
        kernel_size=1,
        strides=1,
        use_bias=True,
        data_format=data_format)(inputs)

    shape = inputs.get_shape().as_list()
    grid_shape = shape[2:4] if data_format == 'channels_first' else shape[1:3]
    
    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 2, 3, 1])
    
    inputs = tf.reshape(inputs, [-1, n_anchors * grid_shape[0] * grid_shape[1], 5 + n_classes])
    
    box_centers, box_shapes, confidence, classes = tf.split(
        inputs, [2, 2, 1, n_classes], axis=-1)

    x = tf.range(grid_shape[0], dtype=tf.float32)
    y = tf.range(grid_shape[1], dtype=tf.float32)
    x_offset, y_offset = tf.meshgrid(x, y)
    x_offset = tf.reshape(x_offset, (-1, 1))
    y_offset = tf.reshape(y_offset, (-1, 1))
    x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
    x_y_offset = tf.tile(x_y_offset, [1, n_anchors])
    x_y_offset = tf.reshape(x_y_offset, [1, -1, 2])
    
    box_centers = tf.nn.sigmoid(box_centers)
    box_centers = (box_centers + x_y_offset) * tf.constant(
        [img_size[1] / grid_shape[1], img_size[0] / grid_shape[0]], 
        dtype=tf.float32)

    anchors = tf.tile(tf.constant(anchors, dtype=tf.float32), [grid_shape[0] * grid_shape[1], 1])
    box_shapes = tf.exp(box_shapes) * anchors

    confidence = tf.nn.sigmoid(confidence)
    classes = tf.nn.sigmoid(classes)

    return tf.concat([box_centers, box_shapes, confidence, classes], axis=-1)

def upsample(inputs, out_shape, data_format):
    """Upsamples to out_shape using nearest neighbor interpolation."""
    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 2, 3, 1])
        new_height = out_shape[3]
        new_width = out_shape[2]
    else:
        new_height = out_shape[2]
        new_width = out_shape[1]

    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width))

    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    return inputs

def build_boxes(inputs):
    """Computes top left and bottom right points of the boxes."""
    center_x, center_y, width, height, confidence, classes = \
        tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)

    top_left_x = center_x - width / 2
    top_left_y = center_y - height / 2
    bottom_right_x = center_x + width / 2
    bottom_right_y = center_y + height / 2

    boxes = tf.concat([top_left_x, top_left_y,
                       bottom_right_x, bottom_right_y,
                       confidence, classes], axis=-1)

    return boxes

def non_max_suppression(inputs, n_classes, max_output_size, iou_threshold,
                        confidence_threshold):
    """Performs non-max suppression separately for each class."""
    batch = tf.unstack(inputs)
    boxes_dicts = []
    for boxes in batch:
        boxes = tf.boolean_mask(boxes, boxes[:, 4] > confidence_threshold)
        classes = tf.argmax(boxes[:, 5:], axis=-1)
        classes = tf.expand_dims(tf.cast(classes, tf.float32), axis=-1)
        boxes = tf.concat([boxes[:, :5], classes], axis=-1)

        boxes_dict = dict()
        for cls in range(n_classes):
            mask = tf.equal(boxes[:, 5], cls)
            mask_shape = mask.get_shape()
            if mask_shape.ndims != 0:
                class_boxes = tf.boolean_mask(boxes, mask)
                boxes_coords, boxes_conf_scores, _ = tf.split(class_boxes,
                                                              [4, 1, -1],
                                                              axis=-1)
                boxes_conf_scores = tf.reshape(boxes_conf_scores, [-1])
                indices = tf.image.non_max_suppression(boxes_coords,
                                                       boxes_conf_scores,
                                                       max_output_size,
                                                       iou_threshold)
                class_boxes = tf.gather(class_boxes, indices)
                boxes_dict[cls] = class_boxes[:, :5]

        boxes_dicts.append(boxes_dict)

    return boxes_dicts

class Yolo_v3:
    """Yolo v3 model class."""
    
    def __init__(self, n_classes, model_size, max_output_size, iou_threshold,
                 confidence_threshold, data_format=None):
        if not data_format:
            if tf.test.is_built_with_cuda():
                data_format = 'channels_first'
            else:
                data_format = 'channels_last'

        self.n_classes = n_classes
        self.model_size = model_size
        self.max_output_size = max_output_size
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.data_format = data_format

    def __call__(self, inputs, training):
        with tf.compat.v1.variable_scope('yolo_v3_model'):
            if self.data_format == 'channels_first':
                inputs = tf.transpose(inputs, [0, 3, 1, 2])

            inputs = inputs / 255

            route1, route2, inputs = darknet53(inputs, training=training,
                                     data_format=self.data_format)

            route, inputs = yolo_convolution_block(
                inputs, filters=512, training=training,
                data_format=self.data_format)
            detect1 = yolo_layer(inputs, n_classes=self.n_classes,
                              anchors=_ANCHORS[6:9],
                              img_size=self.model_size,
                              data_format=self.data_format)

            inputs = conv2d_fixed_padding(route, filters=256, kernel_size=1,
                                        data_format=self.data_format)
            inputs = batch_norm(inputs, training=training,
                              data_format=self.data_format)
            inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)
            upsample_size = route2.get_shape().as_list()
            inputs = upsample(inputs, out_shape=upsample_size,
                            data_format=self.data_format)
            axis = 1 if self.data_format == 'channels_first' else 3
            inputs = tf.concat([inputs, route2], axis=axis)
            route, inputs = yolo_convolution_block(
                inputs, filters=256, training=training,
                data_format=self.data_format)
            detect2 = yolo_layer(inputs, n_classes=self.n_classes,
                              anchors=_ANCHORS[3:6],
                              img_size=self.model_size,
                              data_format=self.data_format)

            inputs = conv2d_fixed_padding(route, filters=128, kernel_size=1,
                                        data_format=self.data_format)
            inputs = batch_norm(inputs, training=training,
                              data_format=self.data_format)
            inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)
            upsample_size = route1.get_shape().as_list()
            inputs = upsample(inputs, out_shape=upsample_size,
                            data_format=self.data_format)
            inputs = tf.concat([inputs, route1], axis=axis)
            route, inputs = yolo_convolution_block(
                inputs, filters=128, training=training,
                data_format=self.data_format)
            detect3 = yolo_layer(inputs, n_classes=self.n_classes,
                              anchors=_ANCHORS[0:3],
                              img_size=self.model_size,
                              data_format=self.data_format)

            inputs = tf.concat([detect1, detect2, detect3], axis=1)
            inputs = build_boxes(inputs)

            boxes_dicts = non_max_suppression(
                inputs, n_classes=self.n_classes,
                max_output_size=self.max_output_size,
                iou_threshold=self.iou_threshold,
                confidence_threshold=self.confidence_threshold)

            return boxes_dicts

def load_images(img_names, model_size):
    """Loads images in a 4D array."""
    imgs = []

    for img_name in img_names:
        img = Image.open(img_name)
        img = img.resize(size=model_size)
        img = np.array(img, dtype=np.float32)
        img = np.expand_dims(img, axis=0)
        imgs.append(img)

    imgs = np.concatenate(imgs)

    return imgs

def load_class_names(file_name):
    """Returns a list of class names read from file_name."""
    with open(file_name, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

def draw_boxes(img_names, boxes_dicts, class_names, model_size):
    """Modified draw_boxes to save images instead of displaying them"""
    colors = ((np.array(color_palette("hls", 80)) * 255)).astype(np.uint8)
    for img_name, boxes_dict in zip(img_names, boxes_dicts):
        img = Image.open(img_name)
        draw = ImageDraw.Draw(img)
        
        # Try to load font or use default
        try:
            font_size = (img.size[0] + img.size[1]) // 100
            font = ImageFont.truetype("arial.ttf", size=font_size)
        except:
            font = ImageFont.load_default()
            
        resize_factor = (img.size[0] / model_size[0], img.size[1] / model_size[1])
        
        for cls in range(len(class_names)):
            boxes = boxes_dict.get(cls, [])
            if len(boxes) > 0:
                color = tuple(map(int, colors[cls]))
                for box in boxes:
                    xy, confidence = box[:4], box[4]
                    xy = [xy[i] * resize_factor[i % 2] for i in range(4)]
                    x0, y0 = xy[0], xy[1]
                    
                    # Draw rectangle
                    draw.rectangle(xy, outline=color, width=3)
                    
                    # Draw label
                    text = f'{class_names[cls]} {confidence*100:.1f}%'
                    # Get text size using textbbox (new method) or textsize (old method)
                    if hasattr(draw, 'textbbox'):
                        text_bbox = draw.textbbox((0, 0), text, font=font)
                        text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
                    else:
                        text_size = draw.textsize(text, font=font)
                    
                    # Draw background rectangle for text
                    draw.rectangle(
                        [x0, y0 - text_size[1], x0 + text_size[0], y0],
                        fill=color)
                    # Draw text
                    draw.text((x0, y0 - text_size[1]), text, fill='white', font=font)
        
        # Save the output image
        output_path = os.path.join("output", os.path.basename(img_name))
        os.makedirs("output", exist_ok=True)
        img.save(output_path)
        print(f"Saved detection results to {output_path}")

def load_weights(variables, file_name):
    """Reshapes and loads official pretrained Yolo weights."""
    with open(file_name, "rb") as f:
        np.fromfile(f, dtype=np.int32, count=5)
        weights = np.fromfile(f, dtype=np.float32)

        assign_ops = []
        ptr = 0

        for i in range(52):
            conv_var = variables[5 * i]
            gamma, beta, mean, variance = variables[5 * i + 1:5 * i + 5]
            batch_norm_vars = [beta, gamma, mean, variance]

            for var in batch_norm_vars:
                shape = var.shape.as_list()
                num_params = np.prod(shape)
                var_weights = weights[ptr:ptr + num_params].reshape(shape)
                ptr += num_params
                assign_ops.append(tf.assign(var, var_weights))

            shape = conv_var.shape.as_list()
            num_params = np.prod(shape)
            var_weights = weights[ptr:ptr + num_params].reshape(
                (shape[3], shape[2], shape[0], shape[1]))
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(tf.assign(conv_var, var_weights))

        ranges = [range(0, 6), range(6, 13), range(13, 20)]
        unnormalized = [6, 13, 20]
        for j in range(3):
            for i in ranges[j]:
                current = 52 * 5 + 5 * i + j * 2
                conv_var = variables[current]
                gamma, beta, mean, variance =  \
                    variables[current + 1:current + 5]
                batch_norm_vars = [beta, gamma, mean, variance]

                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    assign_ops.append(tf.assign(var, var_weights))

                shape = conv_var.shape.as_list()
                num_params = np.prod(shape)
                var_weights = weights[ptr:ptr + num_params].reshape(
                    (shape[3], shape[2], shape[0], shape[1]))
                var_weights = np.transpose(var_weights, (2, 3, 1, 0))
                ptr += num_params
                assign_ops.append(tf.assign(conv_var, var_weights))

            bias = variables[52 * 5 + unnormalized[j] * 5 + j * 2 + 1]
            shape = bias.shape.as_list()
            num_params = np.prod(shape)
            var_weights = weights[ptr:ptr + num_params].reshape(shape)
            ptr += num_params
            assign_ops.append(tf.assign(bias, var_weights))

            conv_var = variables[52 * 5 + unnormalized[j] * 5 + j * 2]
            shape = conv_var.shape.as_list()
            num_params = np.prod(shape)
            var_weights = weights[ptr:ptr + num_params].reshape(
                (shape[3], shape[2], shape[0], shape[1]))
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(tf.assign(conv_var, var_weights))

    return assign_ops

def main():
    # Initialize MLflow
    mlflow.set_tracking_uri("mlruns")  # Local directory to store runs
    mlflow.set_experiment("YOLOv3 Object Detection")
    
    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("model_name", "YOLOv3")
        mlflow.log_param("model_size", "416x416")
        mlflow.log_param("max_output_size", 10)
        mlflow.log_param("iou_threshold", 0.5)
        mlflow.log_param("confidence_threshold", 0.5)
        
        # Log start time
        start_time = datetime.now()
        mlflow.log_param("start_time", start_time.strftime("%Y-%m-%d %H:%M:%S"))
        
        img_names = ["dog.jpg", "office.jpg", "street.jpg"]
        weights_path = "yolov3.weights"
        class_names_path = "coco.names"
        output_dir = "output"
        
        os.makedirs(output_dir, exist_ok=True)

        print("Loading input images...")
        for img_path in img_names:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            img = Image.open(img_path)
            print(f"Loaded: {img_path} ({img.size[0]}x{img.size[1]})")

        class_names = load_class_names(class_names_path)
        n_classes = len(class_names)
        print(f"Loaded {n_classes} class names")
        
        # Log class names count
        mlflow.log_param("n_classes", n_classes)

        print("Initializing YOLOv3 model...")
        model = Yolo_v3(
            n_classes=n_classes,
            model_size=_MODEL_SIZE,
            max_output_size=10,
            iou_threshold=0.5,
            confidence_threshold=0.5
        )

        batch = load_images(img_names, model_size=_MODEL_SIZE)
        batch_size = len(img_names)

        print("Building detection graph...")
        inputs = tf.keras.Input(shape=(416, 416, 3), batch_size=batch_size, dtype=tf.float32)
        detections = model(inputs, training=False)

        # Create Keras model for saving
        keras_model = tf.keras.Model(inputs=inputs, outputs=detections, name='yolo_v3_model')

        model_vars = tf.compat.v1.global_variables(scope='yolo_v3_model')
        assign_ops = load_weights(model_vars, weights_path)

        print("Running object detection...")
        with tf.compat.v1.Session() as sess:
            print("Loading model weights...")
            sess.run(assign_ops)
            
            print("Saving model to .h5 file...")
            keras_model.save('yolov3_model.h5')
            print("Model saved as yolov3_model.h5")
            
            # Log model artifact
            mlflow.log_artifact("yolov3_model.h5")
            
            print("Processing images...")
            detection_result = sess.run(detections, feed_dict={inputs: batch})

        print("Drawing bounding boxes...")
        draw_boxes(img_names, detection_result, class_names, _MODEL_SIZE)
        
        # Log output images
        for img_name in img_names:
            output_path = os.path.join("output", os.path.basename(img_name))
            if os.path.exists(output_path):
                mlflow.log_artifact(output_path)
        
        print("Object detection complete!")
        
        # Log end time and duration
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        mlflow.log_param("end_time", end_time.strftime("%Y-%m-%d %H:%M:%S"))
        mlflow.log_metric("duration_seconds", duration)
        
        print(f"Experiment completed in {duration:.2f} seconds")

if __name__ == "__main__":
    main()