class JumpCycleAnalyzer:
    def __init__(self, df, valid_jump_sets):
        self.df = df
        self.valid_jump_sets = valid_jump_sets
        self.jump_cycles = None

    def find_jump_cycles(self):
        self.jump_cycles = find_jump_cycles(self.df, self.valid_jump_sets)
        return self.jump_cycles

    def filter_similar_cycles(self, similarity_threshold=0.9):
        return filter_highest_pairwise_similar_jump_cycles(
            self.jump_cycles, 
            similarity_threshold=similarity_threshold
        )
