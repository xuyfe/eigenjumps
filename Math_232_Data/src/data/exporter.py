class DataExporter:
    @staticmethod
    def save_to_csv(df, file_path):
        save_filtered_df(df, file_path)

    @staticmethod
    def format_jumps_csv(file_path, jump_cycles):
        return format_jumps_csv(file_path, jump_cycles)
