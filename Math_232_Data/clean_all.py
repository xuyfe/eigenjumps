from convert_fixed_window import convert_and_plot, convert, plot_jump_cycles
import os

print("========== Processing: all files ==========")
for file in os.listdir("data"):
    if file.endswith(".txt"):
        print(f"========== Processing: {file} ==========")
        convert_and_plot(f"data/{file}")