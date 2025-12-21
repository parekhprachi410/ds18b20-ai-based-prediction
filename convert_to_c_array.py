#!/usr/bin/env python3

import os
import tensorflow as tf


def model_to_c_arrays(model, name="temp_predictor"):
    weights = model.get_weights()

    header = f"""#ifndef {name.upper()}_H
#define {name.upper()}_H

#define SEQ_LEN 60
#define INPUT_DIM 1
#define LSTM1_UNITS 64
#define LSTM2_UNITS 32
#define OUTPUT_DIM 1

"""

    source = f'#include "{name}.h"\n\n'

    for i, w in enumerate(weights):
        flat = w.flatten()
        header += f"extern const float weight_{i}[{flat.size}];\n"
        source += f"const float weight_{i}[{flat.size}] = {{\n"

        for j in range(0, flat.size, 8):
            vals = ", ".join(f"{x:.6f}f" for x in flat[j:j+8])
            if j + 8 < flat.size:
                vals += ","
            source += "    " + vals + "\n"

        source += "};\n\n"

    source += "const float* model_weights[] = {\n"
    for i in range(len(weights)):
        source += f"    weight_{i},\n"
    source += "};\n\n"

    source += f"const int num_weight_arrays = {len(weights)};\n"
    source += "const int weight_sizes[] = {\n"
    for w in weights:
        source += f"    {w.size},\n"
    source += "};\n"

    header += """
extern const float* model_weights[];
extern const int num_weight_arrays;
extern const int weight_sizes[];

#endif
"""

    return header, source


def save_files(h, c, name):
    open(name + ".h", "w").write(h)
    open(name + ".c", "w").write(c)
    print("files written:", name + ".h,", name + ".c")


def show_model(model):
    total = 0
    for i, layer in enumerate(model.layers):
        params = sum(w.size for w in layer.get_weights())
        total += params
        print(i, layer.name, params)
    print("total params:", total)


def main():
    model_file = "ds18b20_model.keras"

    if not os.path.exists(model_file):
        print("model not found")
        return

    model = tf.keras.models.load_model(model_file)

    show_model(model)

    h, c = model_to_c_arrays(model)
    save_files(h, c, "temp_predictor")

    total = sum(w.size for w in model.get_weights())
    print("memory ~", round((total * 4) / 1024, 2), "KB")


if __name__ == "__main__":
    main()
