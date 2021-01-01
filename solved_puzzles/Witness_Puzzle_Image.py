import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from collections import deque
import copy


class InvalidPuzzlePositionException(Exception):
    pass


class InvalidColorException(Exception):
    pass


class WitnessPuzzle:
    """
    This class reprensents a state of a puzzle inspired in "separable-colors puzzle" from The Witness

    The state is represented by several matrices:
        (1) _cells stores the colors in each cell, where 0 represents the neutral color
        (i.e., the color that doesn't have to be separated from the others).
        (2) _dots stores the junctions of the cells, where the "snake" can navigate. If
        _cells contains L lines and C columns, then _dots has L+1 lines and C+1 columns.
        (3) _v_seg stores the vertical segments of the puzzle that is occupied by the snake.
        If _v_seg[i][j] equals 1, then the snake is going through that vertical segment.
        _v_seg[i][j] equals 0 otherwise.
        (4) _h_seg is defined analogously for the horizontal segments.

    In addition to the matrices, GameState also stores the line and column of the tip of the snake
    (see _line_tip and _column_tip). The class also contains the starting line and column (_line_init
    and _column_init).

    Currently the state supports 6 colors (see attribute _colors)
    """
    # Possible colors for separable bullets
    _colors = ['b', 'r', 'g', 'c', 'y', 'm']

    def __init__(self, lines=1, columns=1, line_init=0, column_init=0, line_goal=1, column_goal=1, max_lines=8,
                 max_columns=8):
        """
        GameState's constructor. The constructor receives as input the variable lines and columns,
         which specify the number of lines and columns of the puzzle; line_init and column_init speficy
         the entrance position of the snake in the puzzle's grid; line_goal and column_goal speficy
         the exit position of the snake on the grid; max_lines and max_columns are used to embed the puzzle
         into an image of fixed size (see method get_image_representation). The default value of max_lines
         and max_columns is 4. This value has to be increased if working with puzzles larger than 4x4 grid cells.

         The constructor has a step of default values for the variable in case one wants to read a puzzle
         from a text file through method read_state.
        """
        self._v_seg = np.zeros ((lines, columns + 1))
        self._h_seg = np.zeros ((lines + 1, columns))
        self._dots = np.zeros ((lines + 1, columns + 1))
        self._cells = np.zeros ((lines, columns))

        self._column_init = column_init
        self._line_init = line_init
        self._column_goal = column_goal
        self._line_goal = line_goal
        self._lines = lines
        self._columns = columns
        self._line_tip = line_init
        self._column_tip = column_init
        self._max_lines = max_lines
        self._max_columns = max_columns

        # Raises an exception if the initial position of the snake equals is goal position
        if self._column_init == self._column_goal and self._line_init == self._line_goal:
            raise InvalidPuzzlePositionException ('Initial postion of the snake cannot be equal its goal position',
                                                  'Initial: ' +
                                                  str (self._line_init) + ', ' + str (self._column_init) +
                                                  ' Goal: ' + str (self._line_goal) + ', ' + str (self._column_goal))

        # Raises an exception if the initial position of the snake is invalid
        if self._column_init < 0 or self._line_init < 0 or self._column_init > self._columns or self._line_init > self._lines:
            raise InvalidPuzzlePositionException ('Initial position of the snake is invalid',
                                                  str (self._line_init) + ', ' + str (self._column_init))

        # Raises an exception if the goal position of the snake is invalid
        if self._column_goal < 0 or self._line_goal < 0 or self._column_goal > self._columns or self._line_goal > self._lines:
            raise InvalidPuzzlePositionException ('Goal position of the snake is invalid',
                                                  str (self._line_goal) + ', ' + str (self._column_goal))

        # Initializing the tip of the snake
        self._dots[self._line_tip][self._column_tip] = 1

        self._solution_depth = -1

    def __init_structures(self):
        """
        This method initializes the puzzle's structures. We assume that the following attributes were set elsewhere:
        (1) self._lines
        (2) self._columns
        (3) self._line_init
        (4) self._column_init
        (5) self._line_goal
        (6) self._column_goal
        """
        self._v_seg = np.zeros ((self._lines, self._columns + 1))
        self._h_seg = np.zeros ((self._lines + 1, self._columns))
        self._dots = np.zeros ((self._lines + 1, self._columns + 1))
        self._cells = np.zeros ((self._lines, self._columns))

        self._line_tip = self._line_init
        self._column_tip = self._column_init

        self._dots[self._line_tip][self._column_tip] = 1


    def read_state(self, filename):
        """
        Reads a puzzle from a file. It follows the format speficied in method save_state of this class.
        """
        file = open (filename, 'r')

        if '/' in filename:
            self._filename = filename[filename.index ('/') + 1:len (filename)]
        else:
            self._filename = filename
        puzzle = file.read ().split ('\n')
        for s in puzzle:
            if 'Size: ' in s:
                values = s.replace ('Size: ', '').split (' ')
                self._lines = int (values[0])
                self._columns = int (values[1])
                # print("lines", self._lines, " columns", self._columns)
                self._size = [self._lines, self._columns]
            if 'Init: ' in s:
                values = s.replace ('Init: ', '').split (' ')
                self._line_init = int (values[0])
                self._column_init = int (values[1])
            if 'Goal: ' in s:
                values = s.replace ('Goal: ', '').split (' ')
                self._line_goal = int (values[0])
                self._column_goal = int (values[1])
                self.__init_structures ()
            if 'Colors: ' in s:
                values = s.replace ('Colors: |', '').split ('|')
                for t in values:
                    numbers = t.split (' ')
                    self._cells[int (numbers[0])][int (numbers[1])] = int (numbers[2])


    def print_image(self, image):
        for i in range (0, image.shape[2]):
            for j in range (0, image.shape[0]):
                for k in range (0, image.shape[0]):
                    print (image[j][k][i], end=' ')
                print ()
            print ('\n\n')

    def plot(self):
        self.generate_image (False, '')

    def save_figure(self, filename):
        self.generate_image (True, filename)

    def generate_image(self, save_file, filename, figsize=[5, 5]):
        """
        This method plots the state. Several features in this method are hard-coded and might
        need adjustment as one changes the size of the puzzle. For example, the size of the figure is set to be fixed
        to [5, 5] (see below).
        """
        fig = plt.figure (figsize=[5, 5])
        fig.patch.set_facecolor ((1, 1, 1))

        ax = fig.add_subplot (111)

        # draw vertical lines of the grid
        for y in range (self._dots.shape[1]):
            ax.plot ([y, y], [0, self._cells.shape[0]], 'k')
        # draw horizontal lines of the grid
        for x in range (self._dots.shape[0]):
            ax.plot ([0, self._cells.shape[1]], [x, x], 'k')

        # scale the axis area to fill the whole figure
        ax.set_position ([0, 0, 1, 1])

        ax.set_axis_off ()

        ax.set_xlim (-1, np.max (self._dots.shape))
        ax.set_ylim (-1, np.max (self._dots.shape))

        # Draw the vertical segments of the path
        for i in range (self._v_seg.shape[0]):
            for j in range (self._v_seg.shape[1]):
                if self._v_seg[i][j] == 1:
                    ax.plot ([j, j], [i, i + 1], 'r', linewidth=5)

        # Draw the horizontal segments of the path
        for i in range (self._h_seg.shape[0]):
            for j in range (self._h_seg.shape[1]):
                if self._h_seg[i][j] == 1:
                    ax.plot ([j, j + 1], [i, i], 'r', linewidth=5)

        # Draw the separable bullets according to the values in self._cells and self._colors
        offset = 0.5
        for i in range (self._cells.shape[0]):
            for j in range (self._cells.shape[1]):
                if self._cells[i][j] != 0:
                    ax.plot (j + offset, i + offset, 'o', markersize=15, markeredgecolor=(0, 0, 0),
                             markerfacecolor=self._colors[int (self._cells[i][j] - 1)], markeredgewidth=2)

        # Draw the intersection of lines: red for an intersection that belongs to a path and black otherwise
        for i in range (self._dots.shape[0]):
            for j in range (self._dots.shape[1]):
                if self._dots[i][j] != 0:
                    ax.plot (j, i, 'o', markersize=10, markeredgecolor=(0, 0, 0), markerfacecolor='r',
                             markeredgewidth=0)
                else:
                    ax.plot (j, i, 'o', markersize=10, markeredgecolor=(0, 0, 0), markerfacecolor='k',
                             markeredgewidth=0)

        # Draw the entrance of the puzzle in red as it is always on the state's path
        ax.plot (self._column_init - 0.15, self._line_init, '>', markersize=10, markeredgecolor=(0, 0, 0),
                 markerfacecolor='r', markeredgewidth=0)

        column_exit_offset = 0
        line_exit_offset = 0

        if self._column_goal == self._columns:
            column_exit_offset = 0.15
            exit_symbol = '>'
        elif self._column_goal == 0:
            column_exit_offset = -0.15
            exit_symbol = '<'
        elif self._line_goal == self._lines:
            line_exit_offset = 0.15
            exit_symbol = '^'
        else:
            line_exit_offset = -0.15
            exit_symbol = 'v'
        # Draw the exit of the puzzle: red if it is on a path, black otherwise
        if self._dots[self._line_goal][self._column_goal] == 0:
            ax.plot (self._column_goal + column_exit_offset, self._line_goal + line_exit_offset, exit_symbol,
                     markersize=10, markeredgecolor=(0, 0, 0), markerfacecolor='k', markeredgewidth=0)
        else:
            ax.plot (self._column_goal + column_exit_offset, self._line_goal + line_exit_offset, exit_symbol,
                     markersize=10, markeredgecolor=(0, 0, 0), markerfacecolor='r', markeredgewidth=0)

        if save_file:
            plt.savefig (filename)
            # print("filename", filename)
            # plt.show()
            plt.close ()
        else:
            plt.show ()

    def get_image_representation(self):
        """
        Generates an image representation for the puzzle. Currently the method supports 4 colors and includes
        the following channels (third dimension of image): one channel for each color; one channel with 1's
        where is "open" in the grid (this allows learning systems to work with a fixed image size defined
        by max_lines and max_columns); one channel for the current path (cells occupied by the snake);
        one channel for the tip of the snake; one channel for the exit of the puzzle; one channel for the
        entrance of the snake. In total there are 9 different channels.

        Each channel is a matrix with zeros and ones. The image returned is a 3-dimensional numpy array.
        """

        number_of_colors = 4
        channels = 9

        # defining the 3-dimnesional array that will be filled with the puzzle's information
        image = np.zeros ((2 * self._max_lines, 2 * self._max_columns, channels))

        # create one channel for each color i
        for i in range (0, number_of_colors):
            for j in range (0, self._cells.shape[0]):
                for k in range (0, self._cells.shape[1]):
                    if self._cells[j][k] == i:
                        image[2 * j + 1][2 * k + 1][i] = 1
        channel_number = number_of_colors

        # the number_of_colors-th channel specifies the open spaces in the grid
        for j in range (0, 2 * self._lines + 1):
            for k in range (0, 2 * self._columns + 1):
                image[j][k][channel_number] = 1

        # channel for the current path
        channel_number += 1
        for i in range (0, self._v_seg.shape[0]):
            for j in range (0, self._v_seg.shape[1]):
                if self._v_seg[i][j] == 1:
                    image[2 * i][2 * j][channel_number] = 1
                    image[2 * i + 1][2 * j][channel_number] = 1
                    image[2 * i + 2][2 * j][channel_number] = 1

        for i in range (0, self._h_seg.shape[0]):
            for j in range (0, self._h_seg.shape[1]):
                if self._h_seg[i][j] == 1:
                    image[2 * i][2 * j][channel_number] = 1
                    image[2 * i][2 * j + 1][channel_number] = 1
                    image[2 * i][2 * j + 2][channel_number] = 1

        # channel with the tip of the snake
        channel_number += 1
        image[2 * self._line_tip][2 * self._column_tip][channel_number] = 1

        # channel for the exit of the puzzle
        channel_number += 1
        image[2 * self._line_goal][2 * self._column_goal][channel_number] = 1

        # channel for the entrance of the puzzle
        channel_number += 1
        image[2 * self._line_init][2 * self._column_init][channel_number] = 1

        return image