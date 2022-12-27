from pascal_voc_writer import Writer
import os


def create_pascal_label(filename, width, height, bbox):
    # create pascal voc writer (image_path, width, height)
    writer = Writer(f'{filename}_processed.jpg', width, height)

    # add objects (class, xmin, ymin, xmax, ymax)
    writer.addObject('guardrail-damage', *bbox)
    # write to file

    writer.save(os.path.join("debug", f'{filename}_processed.xml'))