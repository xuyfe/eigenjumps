from convert_fixed_window import convert_and_plot, convert, plot_jump_cycles

def main():
    '''
    USAGE: Takes in raw data (txt), converts/extracts jumps, and plots
    RETURNS: None
    CREATES: File called "AnnieGu_jumps_summed.csv" in the data folder
    '''
    convert_and_plot("data/Garrett.txt")

    '''
    USAGE: Takes in raw data (txt) and returns a df of jump cycles
    RETURNS: df of jump cycles
    CREATES: File called "AnnieGu_jumps_summed.csv" in the data folder
    '''
    jump_df = convert("data/Garrett.txt")

    '''
    USAGE: Takes in a df of jump cycles and plots them
    RETURNS: None
    CREATES: Plot of jump cycles – use this to verify that the jump cycles are correct
    '''
    plot_jump_cycles(jump_df)



if __name__ == "__main__":
    main()