class JumpVisualizer:
    def __init__(self, df):
        self.df = df

    def plot_jump_cycles(self, name=None, separate=False):
        plot_jump_cycles(self.df, name=name, separate=separate)

    def plot_sensor(self, sensor_name, peaks=None, jump_cycles=None):
        plot_one_sensor(self.df, sensor_name, peaks, jump_cycles)

    def plot_multiple_sensors(self, sensor_names):
        plot_list_of_sensors(self.df, sensor_names)
