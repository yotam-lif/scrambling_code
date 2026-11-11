"""Create an entirely vector, editable Inkscape version of Figure 3.

Includes the Gaussian fitness function and axis styling from the original plot.
No raster images are embedded. Run with the project's Python environment.
"""
from pathlib import Path
from io import BytesIO
import xml.etree.ElementTree as E
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent


def gaussian_fitness(r1, r2):
    return np.exp(-0.5 * (r1**2 + r2**2))


def style_axis(ax):
    ax.set_xlabel(r"$r_1$")
    ax.set_ylabel(r"$r_2$")
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_position(("outward", 10))
    ax.spines["left"].set_position(("outward", 10))
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.tick_params(width=1.5, length=6, which="major")
    ax.tick_params(width=1.5, length=3, which="minor")

S = 'http://www.w3.org/2000/svg'
I = 'http://www.inkscape.org/namespaces/inkscape'
SO = 'http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd'
for prefix, uri in [('', S), ('inkscape', I), ('sodipodi', SO),
                    ('xlink', 'http://www.w3.org/1999/xlink')]:
    E.register_namespace(prefix, uri)

def el(parent, tag, **attrs):
    return E.SubElement(parent, '{'+S+'}'+tag,
                        {k.replace('_', '-'): str(v) for k, v in attrs.items()})

def path(g, d, fill, **attrs):
    return el(g, 'path', d=d, fill=fill, **attrs)

def label(node, name):
    node.set('{'+I+'}label', name)

def coin(g, x, y, name):
    q = el(g, 'g', id=name)
    label(q, name)
    el(q, 'ellipse', cx=x, cy=y, rx=27.5, ry=27,
       fill='#b67600', stroke='#805000', stroke_width=2.4)
    el(q, 'ellipse', cx=x-3, cy=y-3, rx=24, ry=24, fill='#df9400')
    el(q, 'ellipse', cx=x-8, cy=y-10, rx=5.5, ry=5.3, fill='#efbb57')

def main():
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 14,
    })
    mpl.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['svg.hashsalt'] = 'fig3-editable'
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 10), constrained_layout=True)
    r = np.linspace(-3, 3, 300)
    r1, r2 = np.meshgrid(r, r)
    fitness = gaussian_fitness(r1, r2)
    for ax, panel_label in zip(axes, ['A', 'B']):
        im = ax.imshow(fitness, extent=[-3, 3, -3, 3], origin='lower',
                       cmap='plasma', vmin=0, vmax=1)
        ax.text(-.1, 1.05, panel_label, transform=ax.transAxes,
                fontsize=18, fontweight='bold')
        style_axis(ax)
        ax.set_xticks(range(-3, 4))
        c = fig.colorbar(im, ax=ax, fraction=.046, pad=.02)
        c.set_label('Fitness', rotation=90, labelpad=16, fontsize=16)
        c.ax.yaxis.set_label_position('right')
        c.ax.tick_params(width=1.5, length=6, which='major', labelsize=14)
        c.ax.tick_params(width=1.5, length=3, which='minor')
        c.outline.set_linewidth(1.5)
    stream = BytesIO()
    fig.savefig(stream, format='svg', bbox_inches='tight')
    plt.close(fig)
    root = E.fromstring(stream.getvalue())
    root.set('id', 'fig3-scrambling-explanation')
    defs = root.find('{'+S+'}defs')
    # Smooth plasma interpolation through the sampled Gaussian, as a native SVG
    # gradient. Radius sqrt(18) reaches the corners of the [-3,3]^2 domain.
    grad = el(defs, 'radialGradient', id='gaussian-plasma', cx='.5', cy='.5',
              r=str(np.sqrt(.5)), color_interpolation='sRGB')
    cmap = mpl.colormaps['plasma']
    lut = np.asarray(cmap(np.linspace(0, 1, 256)))[:, :3]
    def color(value):
        rgb = [np.interp(value, np.linspace(0, 1, 256), lut[:, ch]) for ch in range(3)]
        return mpl.colors.to_hex(rgb)
    for offset in np.linspace(0, 1, 513):
        value = float(gaussian_fitness(np.sqrt(18)*offset, 0))
        el(grad, 'stop', offset=offset, stop_color=color(value))
    bar = el(defs, 'linearGradient', id='plasma-colorbar', x1=0, y1=0, x2=0, y2=1)
    for offset in np.linspace(0, 1, 256):
        el(bar, 'stop', offset=offset, stop_color=color(offset))
    plot_boxes = []
    for parent in list(root.iter()):
        for node in list(parent):
            if node.tag != '{'+S+'}image':
                continue
            attrs = {k:v for k,v in node.attrib.items() if not k.endswith('href')}
            heatmap = float(attrs['width']) > 100
            if heatmap:
                plot_boxes.append((float(attrs['x']), -float(attrs['y']),
                                   float(attrs['width']), float(attrs['height'])))
            attrs['fill'] = 'url(#gaussian-plasma)' if heatmap else 'url(#plasma-colorbar)'
            attrs['id'] = ('fitness-background-' if heatmap else 'fitness-colorbar-') + str(len(plot_boxes)) + '-' + node.attrib['id']
            replacement = E.Element('{'+S+'}rect', attrs)
            label(replacement, 'Gaussian fitness — editable radial gradient' if heatmap
                  else 'Fitness color scale — editable linear gradient')
            index = list(parent).index(node)
            parent.remove(node)
            parent.insert(index, replacement)
    for node in root.iter():
        id = node.get('id', '')
        if id.startswith('axes_'):
            node.set('{'+I+'}groupmode', 'layer')
            label(node, {'axes_1':'A — background, axes and text',
                         'axes_2':'B — background, axes and text',
                         'axes_3':'A — fitness scale', 'axes_4':'B — fitness scale'}[id])
    grad = el(defs, 'linearGradient', id='arc-fade', gradientUnits='userSpaceOnUse',
              x1=330, y1=0, x2=640, y2=0)
    for offset, opacity in [('0','0'),('.7','.95'),('1','1')]:
        el(grad, 'stop', offset=offset, stop_color='white', stop_opacity=opacity)
    def panel(name, id, box, yref):
        x,y,w,h = box
        g=el(root,'g',id=id,transform=f'translate({x} {y}) scale({w/611} {h/598}) translate(-149 {-yref})')
        g.set('{'+I+'}groupmode','layer');label(g,name)
        return g
    A=panel('A — scrambling and mutation effects','panel-a-overlays',plot_boxes[0],75)
    B=panel('B — contours and mutation effects','panel-b-overlays',plot_boxes[1],863)
    path(A,'M 333,195 C 382,135 444,109 511,118 C 613,128 701,214 705,315 C 709,381 681,441 646,477','none',stroke='url(#arc-fade)',stroke_width=7,stroke_linecap='round')
    path(A,'M 625,495 L 638,457 L 663,485 Z','white')
    coin(A,319,214,'A-starting-genotype');coin(A,588,512,'A-scrambled-genotype')
    green='#62b33e';red='#c83c31'
    path(A,'M 352,246 L 399,295 L 409,284 L 420,322 L 383,309 L 394,299 Z',green)
    path(A,'M 622,545 L 670,594 L 679,583 L 689,622 L 652,607 L 663,598 Z',red)
    el(B,'ellipse',cx=455,cy=1156,rx=243.5,ry=238.5,fill='none',stroke='white',stroke_width=2.6,stroke_dasharray='15 15')
    el(B,'ellipse',cx=461,cy=1162,rx=82,ry=80.5,fill='none',stroke='white',stroke_width=2.6,stroke_dasharray='15 15')
    path(B,'M 325,975 L 504,972 L 504,959 L 534,975 L 504,991 L 504,978 Z',green)
    path(B,'M 328,998 L 391,1041 L 397,1032 L 408,1056 L 381,1051 L 387,1045 Z',green)
    path(B,'M 274,1005 L 200,1086 L 192,1077 L 182,1109 L 212,1096 L 204,1090 Z',red)
    path(B,'M 440,1102 L 621,1099 L 621,1086 L 651,1102 L 621,1118 L 621,1105 Z',red)
    path(B,'M 444,1125 L 507,1168 L 512,1158 L 522,1182 L 496,1177 L 503,1171 Z',green)
    path(B,'M 390,1123 L 307,1207 L 299,1197 L 289,1228 L 319,1215 L 311,1210 Z',red)
    coin(B,295,981,'B-low-fitness-genotype');coin(B,410,1106,'B-high-fitness-genotype')
    # Label each individual editable shape for Inkscape's Objects panel.
    for panel_node in [A,B]:
        for index,node in enumerate(panel_node):
            if node.get('fill') in [green, red]:
                node.set('stroke', '#ffffff')
                node.set('stroke-width', '0.6')
                node.set('stroke-linejoin', 'round')
            if not node.get('id'):node.set('id',f'{panel_node.get("id")}-shape-{index+1}')
            if not node.get('{'+I+'}label'):
                label(node, 'Fitness contour' if node.tag.endswith('ellipse') else 'Mutation arrow' if node.get('fill') in [green,red] else 'Scrambling arrow')
    assert not list(root.iter('{'+S+'}image'))
    output = SCRIPT_DIR / 'ext_svg' / 'fig3_scrambling_explanation.svg'
    output.parent.mkdir(parents=True, exist_ok=True)
    E.ElementTree(root).write(output,encoding='utf-8',xml_declaration=True)
    print(output)

if __name__ == '__main__':
    main()
