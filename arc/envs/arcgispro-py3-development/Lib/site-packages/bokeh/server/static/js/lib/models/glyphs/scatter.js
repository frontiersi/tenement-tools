import { Marker, MarkerView } from "./marker";
import { marker_funcs } from "./defs";
import { MarkerGL } from "./webgl/markers";
import * as p from "../../core/properties";
export class ScatterView extends MarkerView {
    _init_webgl() {
        const { webgl } = this.renderer.plot_view.canvas_view;
        if (webgl != null) {
            const marker_types = new Set(this.marker);
            if (marker_types.size == 1) {
                const [marker_type] = [...marker_types];
                if (MarkerGL.is_supported(marker_type)) {
                    const { glglyph } = this;
                    if (glglyph == null || glglyph.marker_type != marker_type) {
                        this.glglyph = new MarkerGL(webgl.gl, this, marker_type);
                        return;
                    }
                }
            }
        }
        delete this.glglyph;
    }
    _set_data(indices) {
        super._set_data(indices);
        this._init_webgl();
    }
    _render(ctx, indices, data) {
        const { sx, sy, size, angle, marker } = data ?? this;
        for (const i of indices) {
            const sx_i = sx[i];
            const sy_i = sy[i];
            const size_i = size.get(i);
            const angle_i = angle.get(i);
            const marker_i = marker.get(i);
            if (isNaN(sx_i + sy_i + size_i + angle_i) || marker_i == null)
                continue;
            const r = size_i / 2;
            ctx.beginPath();
            ctx.translate(sx_i, sy_i);
            if (angle_i)
                ctx.rotate(angle_i);
            marker_funcs[marker_i](ctx, i, r, this.visuals);
            if (angle_i)
                ctx.rotate(-angle_i);
            ctx.translate(-sx_i, -sy_i);
        }
    }
    draw_legend_for_index(ctx, { x0, x1, y0, y1 }, index) {
        const n = index + 1;
        const marker = this.marker.get(index);
        const args = {
            ...this._get_legend_args({ x0, x1, y0, y1 }, index),
            marker: new p.UniformScalar(marker, n),
        };
        this._render(ctx, [index], args);
    }
}
ScatterView.__name__ = "ScatterView";
export class Scatter extends Marker {
    constructor(attrs) {
        super(attrs);
    }
    static init_Scatter() {
        this.prototype.default_view = ScatterView;
        this.define(() => ({
            marker: [p.MarkerSpec, { value: "circle" }],
        }));
    }
}
Scatter.__name__ = "Scatter";
Scatter.init_Scatter();
//# sourceMappingURL=scatter.js.map