from convert_fixed_window import convert_and_plot, convert, plot_jump_cycles
import os

print("========== Processing: all files ==========")
for file in os.listdir("data"):
    ## exlcude "_max_similarities"
    if file.endswith(".txt") and "_max_similarities" not in file:
        print(f"========== Processing: {file} ==========")
        convert_and_plot(f"data/{file}")