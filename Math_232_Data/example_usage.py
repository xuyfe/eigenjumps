from convert_fixed_window import convert_and_plot, convert, plot_jump_cycles

def main():
    '''
    USAGE: Takes in raw data (txt), converts/extracts jumps, and plots
    '''
    convert_and_plot("data/AnnieGu.txt")

    '''
    USAGE: Takes in raw data (txt) and returns a df of jump cycles
    '''
    jump_df = convert("data/AnnieGu.txt")

    '''
    USAGE: Takes in a df of jump cycles and plots them
    Note: use this to visually verify that the jump cycles are correct
    '''
    plot_jump_cycles(jump_df)




if __name__ == "__main__":
    main()