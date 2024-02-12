import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from BMI.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.applications.ResNet101V2(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, output_variables, learning_rate):
        flatten_in = tf.keras.layers.Flatten()(model.output)
        outputs = []
        for output_var in output_variables:
            if output_var["TYPE"] == "linear":
                outputs.append(tf.keras.layers.Dense(units=1, name=output_var["NAME"])(flatten_in))
            elif output_var["TYPE"] == "binary":
                outputs.append(tf.keras.layers.Dense(units=1, activation="sigmoid", name=output_var["NAME"])(flatten_in))

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=outputs
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss={
                output_var["NAME"]: "mean_squared_error" if output_var["TYPE"] == "linear" else "binary_crossentropy"
                for output_var in output_variables
            },
            metrics={output_var["NAME"]: "accuracy" for output_var in output_variables if output_var["TYPE"] == "binary"}
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            output_variables=self.config.params_output_variables,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
