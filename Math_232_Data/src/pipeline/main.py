class JumpAnalysisPipeline:
    def __init__(self, file_path):
        self.file_path = file_path
        self.loader = DataLoader(file_path)
        self.analyzer = None
        self.visualizer = None

    def run(self):
        # Load and clean data
        df, valid_jump_sets = self.loader.load_data()
        df, valid_jump_sets = self.loader.clean_data()

        # Analyze jump cycles
        self.analyzer = JumpCycleAnalyzer(df, valid_jump_sets)
        jump_cycles = self.analyzer.find_jump_cycles()
        filtered_df, similarities = self.analyzer.filter_similar_cycles()

        # Visualize results
        self.visualizer = JumpVisualizer(filtered_df)
        self.visualizer.plot_jump_cycles()

        # Export results
        exporter = DataExporter()
        exporter.save_to_csv(filtered_df, self.file_path.replace(".txt", "_filtered.csv"))

        return filtered_df, similarities
