import sys
from PIL import Image
import argparse


def concat1():
  parser = argparse.ArgumentParser(description='Concatenate images horizontally')
  parser.add_argument('images', metavar='image', type=str, nargs='+')
  parser.add_argument('--output', '-o', type=str, default='output.jpg')
  args = parser.parse_args()

  images = [Image.open(x) for x in args.images]
  widths, heights = zip(*(i.size for i in images))

  res_width = max(widths)
  res_height = sum(heights)

  new_im = Image.new('RGB', (res_width, res_height))

  y_offset = 0
  for im in images:
    new_im.paste(im, (0, y_offset))
    y_offset += im.size[1]

  new_im.save(args.output)

def concat2():
  imgs = {}
  max_width = 0
  max_height = 0
  for ch in "ABCD":
    imgs[ch] = {}
    for fi in range(1, 5):
      imgs[ch][fi] = Image.open(f"img/event_count_1ms_comp_{ch}{fi}.png")
      max_width = max(max_width, imgs[ch][fi].size[0])
      max_height = max(max_height, imgs[ch][fi].size[1])

  new_im = Image.new('RGB', (max_width*4, max_height*4))

  for ch_i, ch in enumerate("ABCD"):
    for fi in range(1, 5):
      new_im.paste(imgs[ch][fi], (ch_i * max_width, (fi-1) * max_height))

  new_im.save('output.png')

# if __name__ == '__main__':
#   concat1()
#   concat2()
