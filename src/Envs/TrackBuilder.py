import os
import pygame
import numpy as np
import pickle

from src.Envs.consts import *

DEFAULT_TRACKS_FOLDER_PATH = '.\\Tracks'


class CoordinateTransformer:
    def __init__(self, display):
        display_width, display_height = display.get_size()
        self._display_width = display_width
        self._display_height = display_height

    def cartesian_to_screen(self, point):
        # Translate cartesian normalized coordinates to pygame screen coordinates
        x = int(point[0] * self._display_width)
        y = int((1 - point[1]) * self._display_height)
        return [x, y]

    def screen_to_cartesian(self, point):
        x = float(point[0]) / self._display_width
        y = 1 - float(point[1]) / self._display_height
        return [x, y]


class Vertex:
    def __init__(self, position):
        self.position = position
        self.lines = []


class Line:
    def __init__(self, v1, v2, line_id):
        self.line_id = line_id
        v1.lines.append(self)
        v2.lines.append(self)
        self.v1 = v1
        self.v2 = v2
        self.m = (v2.position[1] - v1.position[1]) / (v2.position[0] - v1.position[0] + 1e-20)
        self.n = v1.position[1] - self.m * v1.position[0]


class Loop:
    def __init__(self, start_id=0):
        self.start_id = start_id
        self.vert_list = []
        self.line_list = []
        self.done = False
        self.line_id = start_id

    def build_loop_from_vertex_position_list(self, vertex_list):
        for vertex in vertex_list:
            self.add_new_vertex(vertex)
        self.close_loop()

    def add_new_vertex(self, new_vertex_position):
        self.line_id += 1
        new_vertex = Vertex(new_vertex_position)
        self.vert_list.append(new_vertex)
        if len(self.vert_list) > 1:
            self.line_list.append(Line(self.vert_list[-2], self.vert_list[-1], self.line_id))

    def close_loop(self):
        # connect last point to the first, finish the loop
        if len(self.vert_list) > 0:
            self.line_id += 1
            self.line_list.append(Line(self.vert_list[-1], self.vert_list[0], self.line_id))
        self.done = True


class Track:

    def __init__(self, name, starting_position=None, starting_direction=None, vertex_coodinate_lists=None):
        if vertex_coodinate_lists is None:
            vertex_coodinate_lists = list()
        self.vertex_coodinate_lists = vertex_coodinate_lists
        self.name = name
        self.starting_position = starting_position
        self.starting_direction = starting_direction
        self.loops = list()
        self.vertex_lists = list()
        self.line_lists = list()
        self._build_loops(vertex_coodinate_lists)

        self._loop_count = len(self.vertex_lists)
        self.coordinate_transformer = None

    def _build_loops(self, vertex_lists):
        for i, vertex_list in enumerate(vertex_lists):
            new_loop = Loop(10 ** i)
            new_loop.build_loop_from_vertex_position_list(vertex_list)
            self.loops.append(new_loop)
            self.vertex_lists.append(new_loop.vert_list)
            self.line_lists.append(new_loop.line_list)

    def render(self, game_display):
        if self.coordinate_transformer is None:
            self.coordinate_transformer = CoordinateTransformer(game_display)
        self._loop_count = len(self.vertex_lists)
        if len(self.vertex_lists) > 0:

            for loop in range(self._loop_count):
                for vertex in self.vertex_lists[loop]:
                    pygame.draw.circle(game_display, COLORS_DICT['blue'],
                                       self.coordinate_transformer.cartesian_to_screen(vertex.position),
                                       6)
                for line in self.line_lists[loop]:
                    pygame.draw.line(game_display, COLORS_DICT['blue'],
                                     self.coordinate_transformer.cartesian_to_screen(line.v1.position),
                                     self.coordinate_transformer.cartesian_to_screen(line.v2.position), 4)

    def save(self, folder_path):
        pkl_file_path = os.path.join(folder_path, self.name + '.pkl')
        image_file_path = os.path.join(folder_path, self.name + '.png')
        track_parameters_dict = {'name': self.name,
                                 'starting_position': self.starting_position,
                                 'starting_direction': self.starting_direction,
                                 'vertex_coodinate_lists': self.vertex_coodinate_lists,
                                 }
        with open(pkl_file_path, 'wb') as file:
            pickle.dump(track_parameters_dict, file)

        print('Saving track image')
        pygame.init()
        display_width = 1200
        display_height = 750
        game_display = pygame.display.set_mode((display_width, display_height))

        starting_point = StartingPoint(self.starting_position, self.starting_direction)
        starting_point.render(game_display)
        self.render(game_display)
        pygame.image.save(game_display, image_file_path)

    @staticmethod
    def load(parameters_file_path):
        with open(parameters_file_path, 'rb') as file:
            track_parameters_dict = pickle.load(file)
        return Track(**track_parameters_dict)


class StartingPoint:
    def __init__(self, pos=None, direction=None):
        self.pos = pos
        self.dir = direction
        self.coordinate_transformer = None

    def pick(self, game_display, mouse):
        if self.coordinate_transformer is None:
            self.coordinate_transformer = CoordinateTransformer(game_display)
        mouse_pos = pygame.mouse.get_pos()
        pygame.draw.circle(game_display, (0, 100, 0), mouse_pos, 6)
        if self.pos is None:
            if mouse[0]:
                self.pos = self.coordinate_transformer.screen_to_cartesian(pygame.mouse.get_pos())
        else:
            pygame.draw.circle(game_display, COLORS_DICT['green'],
                               self.coordinate_transformer.cartesian_to_screen(self.pos), 6)
            if mouse[0]:
                p = self.coordinate_transformer.screen_to_cartesian(pygame.mouse.get_pos())
                self.dir = 180 / np.pi * np.arctan2(p[1] - self.pos[1], p[0] - self.pos[0])

    def render(self, game_display):
        if self.coordinate_transformer is None:
            self.coordinate_transformer = CoordinateTransformer(game_display)
        if self.dir is not None:
            pygame.draw.circle(game_display, COLORS_DICT['red'], self.coordinate_transformer.cartesian_to_screen(
                self.pos + 0.01 * np.array([np.cos(self.dir * np.pi / 180), np.sin(self.dir * np.pi / 180)])), 4)
        if self.pos is not None:
            pygame.draw.circle(game_display, COLORS_DICT['green'],
                               self.coordinate_transformer.cartesian_to_screen(self.pos), 6)


def build_track():
    track_name = input("Track name:")

    # Display
    pygame.init()
    pygame.font.init()
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (200, 50)
    display_width = 1200
    display_height = 750
    game_display = pygame.display.set_mode((display_width, display_height))
    pygame.display.set_caption("Build Runner Track GUI")
    clock = pygame.time.Clock()

    s2c = CoordinateTransformer(game_display).screen_to_cartesian

    loop_count = 2
    # loops = [Loop(10 ** l) for l in range(loop_count)]
    loop_vertex_lists = [[] for _ in range(loop_count)]
    starting_point = StartingPoint()
    track = Track(track_name)

    delay = 0
    loop_id = 0
    exit_game = False
    while not exit_game:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit_game = True
        if delay < 10:
            mouse = (0, 0, 0)
            delay += 1
        else:
            mouse = pygame.mouse.get_pressed()
            if sum(mouse) > 0:  # if any buttons are clicked, reset delay
                delay = 0

        # Create inner and outer loops for the track
        if loop_id < loop_count:

            if mouse[0]:
                mouse_pos = pygame.mouse.get_pos()
                mouse_pos_cartesian = s2c(mouse_pos)
                # current_loop.add_new_vertex(mouse_pos_cartesian)
                loop_vertex_lists[loop_id].append(mouse_pos_cartesian)
            if mouse[2]:
                # current_loop.close_loop()
                loop_id += 1

        # Create starting point and direction
        else:
            if starting_point.dir is None:
                starting_point.pick(game_display, mouse)
            else:
                exit_game = True

        track = Track(track_name, vertex_coodinate_lists=loop_vertex_lists)
        # Render
        game_display.fill((0, 0, 0))
        track.render(game_display)
        starting_point.render(game_display)
        pygame.display.update()
        clock.tick(60)

    track.starting_position, track.starting_direction = starting_point.pos, starting_point.dir
    pygame.quit()
    return track


def load_and_render(track_file_path):
    track = Track.load(track_file_path)
    starting_point = StartingPoint(track.starting_position, track.starting_direction)

    pygame.init()
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (200, 50)
    display_width = 1200
    display_height = 750
    game_display = pygame.display.set_mode((display_width, display_height))
    pygame.display.set_caption("RunnerTrack")

    exit_game = False
    while not exit_game:
        # In case quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit_game = True
        starting_point.render(game_display)
        track.render(game_display)
        pygame.display.update()
    pygame.quit()


def build_round_track(vertices_inside_n, vertices_outside_n):
    # inside bound
    d_angle = (360 / vertices_inside_n) * np.pi / 180
    radius_inside = 0.25
    vert_in_list = []
    for i in range(vertices_inside_n):
        xi = 0.5 + radius_inside * np.cos(d_angle * i)
        yi = 0.5 + radius_inside * np.sin(d_angle * i)
        vert_in_list.append((xi, yi))
    # outside bound
    radius_outside = 0.4
    d_angle = (360 / vertices_outside_n) * np.pi / 180
    vert_out_list = []
    for i in range(vertices_outside_n):
        xi = 0.5 + radius_outside * np.cos(d_angle * i)
        yi = 0.5 + radius_outside * np.sin(d_angle * i)
        vert_out_list.append((xi, yi))

    vert_lists = [vert_in_list, vert_out_list]
    sp_pos = [0.5 - 0.25 - 0.15 / 2, 0.5]
    sp_dir = 90
    track_name = f'round__v_inside_{vertices_inside_n}__v_outside_{vertices_outside_n}'
    round_track = Track(track_name, sp_pos, sp_dir, vert_lists)
    return round_track


def main():
    # Run game
    # track = build_track()
    # track.save(DEFAULT_TRACKS_FOLDER_PATH)
    track = build_round_track(10, 10)
    track.save(DEFAULT_TRACKS_FOLDER_PATH)

    # track_path = '.\\Tracks\\round__v_inside_6__v_outside_6.pkl'
    # load_and_render(track_path)


if __name__ == "__main__":
    main()
