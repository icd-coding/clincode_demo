import matplotlib.pyplot as plt
from matplotlib import colors

def colorize(words, weights):
    cmap=plt.cm.Reds
    template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
    colored_string = ''
    norm = colors.Normalize(weights.min(), weights.max())
    for word, color in zip(words, weights):
        color = colors.rgb2hex(cmap(norm(color))[:3])
        colored_string += template.format(color, '&nbsp' + word + '&nbsp' + ' ')
    return colored_string
