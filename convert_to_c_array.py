#!/usr/bin/env python3
"""
convert_to_c_array.py - Fixed version for TensorFlow compatibility
"""

import tensorflow as tf
import numpy as np
import os

def model_to_c_arrays(model, array_name="lstm_model"):
    """Convert TensorFlow model to C arrays for ESP32"""
    
    weights = model.get_weights()
    header_content = ""
    source_content = ""
    
    print(f"Model has {len(weights)} weight arrays")
    
    # Create header file
    header_content = f"""#ifndef {array_name.upper()}_H
#define {array_name.upper()}_H

// LSTM Model Configuration
#define SEQ_LEN 60
#define INPUT_DIM 1
#define LSTM1_UNITS 64
#define LSTM2_UNITS 32
#define OUTPUT_DIM 1

// Model weights and biases
"""
    
    # Create source file
    source_content = f"""#include "{array_name}.h"

"""
    
    # Process each weight array
    for i, weight in enumerate(weights):
        flat_weights = weight.flatten()
        array_name_clean = f"weight_{i}"
        
        # Add to header
        header_content += f"extern const float {array_name_clean}[{len(flat_weights)}];\n"
        
        # Add to source with proper formatting
        source_content += f"const float {array_name_clean}[{len(flat_weights)}] = {{\n"
        
        # Write in chunks of 8 for readability
        for j in range(0, len(flat_weights), 8):
            chunk = flat_weights[j:j+8]
            line = "    " + ", ".join([f"{w:.6f}f" for w in chunk])
            if j + 8 < len(flat_weights):
                line += ","
            source_content += line + "\n"
        
        source_content += "};\n\n"
    
    # Add weight pointers and metadata
    source_content += f"""// Weight array pointers
const float* model_weights[] = {{
"""
    for i in range(len(weights)):
        source_content += f"    weight_{i}"
        if i < len(weights) - 1:
            source_content += ","
        source_content += "\n"
    
    source_content += f"""}};

const int num_weight_arrays = {len(weights)};
const int weight_sizes[] = {{
"""
    for i, weight in enumerate(weights):
        source_content += f"    {weight.size}"
        if i < len(weights) - 1:
            source_content += ","
        source_content += f"  // weight_{i} shape {weight.shape}\n"
    
    source_content += "};\n"
    
    header_content += f"""
// Weight array pointers
extern const float* model_weights[];
extern const int num_weight_arrays;
extern const int weight_sizes[];

#endif // {array_name.upper()}_H
"""
    
    return header_content, source_content

def save_c_files(header_content, source_content, model_name="lstm_model"):
    """Save C header and source files"""
    
    with open(f"{model_name}.h", "w") as f:
        f.write(header_content)
    
    with open(f"{model_name}.c", "w") as f:
        f.write(source_content)
    
    print(f"✓ Saved {model_name}.h and {model_name}.c")

def print_model_summary(model):
    """Print model architecture summary - FIXED for TF compatibility"""
    print("\n" + "="*50)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*50)
    
    total_params = 0
    for i, layer in enumerate(model.layers):
        layer_type = type(layer).__name__
        weights = layer.get_weights()
        layer_params = sum([w.size for w in weights])
        total_params += layer_params
        
        print(f"Layer {i}: {layer_type} - {layer.name}")
        
        # FIX: Use layer.output instead of layer.output_shape for newer TF versions
        try:
            if hasattr(layer, 'output_shape'):
                print(f"  Output shape: {layer.output_shape}")
            else:
                # For newer TF versions, get output shape from the layer's output
                output_shape = layer.output.shape
                print(f"  Output shape: {output_shape}")
        except:
            print(f"  Output shape: Could not determine")
        
        print(f"  Parameters: {layer_params}")
        for j, w in enumerate(weights):
            print(f"    Weight {j} shape: {w.shape}")
        print()
    
    print(f"TOTAL PARAMETERS: {total_params}")
    print("="*50)

def get_layer_info(model):
    """Get detailed layer information for LSTM implementation"""
    print("\n" + "="*50)
    print("LAYER INFORMATION FOR LSTM IMPLEMENTATION")
    print("="*50)
    
    weight_index = 0
    layer_info = []
    
    for i, layer in enumerate(model.layers):
        layer_type = type(layer).__name__
        weights = layer.get_weights()
        
        info = {
            'index': i,
            'type': layer_type,
            'name': layer.name,
            'weights': []
        }
        
        print(f"Layer {i}: {layer_type} - {layer.name}")
        
        for j, weight in enumerate(weights):
            weight_info = {
                'index': weight_index,
                'shape': weight.shape,
                'size': weight.size
            }
            info['weights'].append(weight_info)
            
            print(f"  Weight {weight_index}: shape {weight.shape}")
            weight_index += 1
        
        layer_info.append(info)
        print()
    
    return layer_info

def main():
    # Load your trained model
    model_path = 'ds18b20_model.keras'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Available files:")
        for f in os.listdir('.'):
            print(f"  - {f}")
        return
    
    print("Loading trained model...")
    model = tf.keras.models.load_model(model_path)
    
    # Print model summary (with fix)
    print_model_summary(model)
    
    # Get detailed layer information
    layer_info = get_layer_info(model)
    
    # Convert to C arrays
    print("\nConverting model to C arrays...")
    header_content, source_content = model_to_c_arrays(model, "temp_predictor")
    
    # Save C files
    save_c_files(header_content, source_content, "temp_predictor")
    
    # Print file sizes
    print(f"\nGenerated files:")
    print(f"  - temp_predictor.h: {os.path.getsize('temp_predictor.h')} bytes")
    print(f"  - temp_predictor.c: {os.path.getsize('temp_predictor.c')} bytes")
    
    # Calculate total memory usage
    total_floats = 0
    for weight in model.get_weights():
        total_floats += weight.size
    
    total_memory = total_floats * 4  # 4 bytes per float32
    print(f"\nMemory requirements:")
    print(f"  - Total parameters: {total_floats}")
    print(f"  - Estimated RAM: {total_memory / 1024:.2f} KB")
    print(f"  - Estimated Flash: {total_memory / 1024:.2f} KB")
    
    # Print weight mapping for LSTM implementation
    print(f"\nWEIGHT MAPPING FOR LSTM CODE:")
    print("Use these indices in your lstm_inference.cpp:")
    weight_index = 0
    for layer in layer_info:
        for weight in layer['weights']:
            print(f"  weight_{weight_index} -> Layer '{layer['name']}' shape {weight['shape']}")
            weight_index += 1

if __name__ == '__main__':
    main()