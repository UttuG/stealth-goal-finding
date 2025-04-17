    """Update main grid visualization"""
        self.main_ax.clear()
        self.main_ax.set_xticks([])
        self.main_ax.set_yticks([])
        self.main_ax.set_title("Main Grid")
        self.main_ax.set_xlim(0, self.grid_size)
        self.main_ax.set_ylim(0, self.grid_size)
        self.main_ax.set_aspect('equal', adjustable='box')

        # Draw grid lines
        for i in range(self.grid_size + 1):
            self.main_ax.axhline(i, color='black', linewidth=0.5)
            self.main_ax.axvline(i, color='black', linewidth=0.5)

        # Draw all cells
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                value = self.grid_matrix[i, j]
                color = 'white'
                if value == 1: color = 'blue'
                elif value == 2: color = 'green'
                elif value == 3: color = 'black'
                elif value == 4: color = 'yellow'
                rect = Rectangle((j, i), 1, 1, facecolor=color, edgecolor='black', linewidth=0.5)
                self.main_ax.add_patch(rect)

        self.main_ax.figure.canvas.draw_idle()