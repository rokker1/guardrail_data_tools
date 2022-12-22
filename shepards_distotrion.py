from collections import namedtuple
from wand.color import Color
from wand.image import Image

Point = namedtuple('Point', ['x', 'y', 'i', 'j'])


def shepards_distortion(filename, x0, y0, x1, x2, power):
    with Image(filename=filename) as img:
        width = img.width
        height = img.height
        img.artifacts['shepards:power'] = power
        alpha = Point(0, 0, 0, 0)
        beta = Point(width, 0, width, 0)
        gamma = Point(0, height, 0, height)
        delta = Point(width, height, width, height)
        f = Point(x0, y0, x1, x2)
        args = (
            *f,
            *alpha,
            *beta,
            *gamma,
            *delta,
        )
        img.distort('shepards', args)
        img.format = 'jpeg'
        img.save(filename=f'{filename}_processed.jpg')