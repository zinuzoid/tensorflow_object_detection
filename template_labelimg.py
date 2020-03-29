import os

TRAIN_DIR = '/Users/jimjm/jim/workspace/tensorflow/training/images/test'
ANNOTATIONS_DIR = TRAIN_DIR


def get_filename(filepath):
    filename, _ = os.path.splitext(filepath)
    return filename


def get_fileext(filepath):
    _, fileext = os.path.splitext(filepath)
    return fileext


def main():
    annotations = os.listdir(ANNOTATIONS_DIR)
    annotations = list(filter(lambda s: get_fileext(s) == '.xml', annotations))
    annotations.sort()
    print(annotations)
    last_annotation = list(annotations)[-1]

    images = os.listdir(TRAIN_DIR)
    images = list(filter(lambda s: get_fileext(s) == '.jpg', images))
    images.sort()
    print(images)
    last_image_filename = get_filename(last_annotation) + '.jpg'
    last_image_index = images.index(last_image_filename)
    next_image_filename = images[last_image_index + 1]

    next_annotation_filename = get_filename(next_image_filename) + '.xml' \
 \
    with open(ANNOTATIONS_DIR + '/' + last_annotation) as f:
        lines = f.readlines()
        if len(lines) == 0:
            raise NotImplemented('line empty')
        lines[0] = '<annotation>'
        lines[2] = lines[2].replace(last_image_filename, next_image_filename)
        lines[3] = lines[3].replace(last_image_filename, next_image_filename)

    next_annotation_filepath = ANNOTATIONS_DIR + '/' + next_annotation_filename
    if os.path.exists(next_annotation_filepath):
        raise NotImplemented('file already existed')

    with open(next_annotation_filepath, 'w') as f:
        f.writelines(lines)

    print(next_annotation_filename)
    print(next_image_filename)


main()
