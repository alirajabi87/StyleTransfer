import tensorflow as tf


class StyleContentModel(tf.keras.Model):
    def __init__(self, style_layers, content_layers, vgg_layers):
        super(StyleContentModel, self).__init__()
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.vgg = vgg_layers(style_layers + content_layers)
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def gram_matrix(self, input_tensor):
        """
        Calculate the means and correlations across the different feature maps.
        By taking the outer product of the feature vector with itself at each
        location, and averaging that outer product over all locations.
        """
        results = tf.linalg.einsum('bijc, bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return results / (num_locations)

    def call(self, inputs):
        "Expects float input in the range of [0, 1]"
        if tf.math.reduce_max(inputs) <= 1:
            inputs = inputs * 255.0

        preprocessed_input1 = tf.keras.applications.vgg16.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input1)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [self.gram_matrix(style_output) for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value
                      for style_name, value in zip(self.style_layers, style_outputs)}
        return {'content': content_dict, 'style': style_dict}


    def high_pass_x_y(self, image):
        x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
        y_var = image[:, 1:, :, :] - image[:, :-1, :, :]
        return x_var, y_var