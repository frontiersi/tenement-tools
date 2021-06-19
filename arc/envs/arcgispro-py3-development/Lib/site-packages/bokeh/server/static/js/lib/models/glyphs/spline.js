import { XYGlyph, XYGlyphView } from "./xy_glyph";
import * as mixins from "../../core/property_mixins";
import { catmullrom_spline } from "../../core/util/interpolation";
export class SplineView extends XYGlyphView {
    _set_data() {
        const { tension, closed } = this.model;
        [this._xt, this._yt] = catmullrom_spline(this._x, this._y, 20, tension, closed);
    }
    _map_data() {
        const { x_scale, y_scale } = this.renderer.coordinates;
        this.sxt = x_scale.v_compute(this._xt);
        this.syt = y_scale.v_compute(this._yt);
    }
    _render(ctx, _indices, data) {
        const { sxt: sx, syt: sy } = data ?? this;
        this.visuals.line.set_value(ctx);
        const n = sx.length;
        for (let j = 0; j < n; j++) {
            if (j == 0) {
                ctx.beginPath();
                ctx.moveTo(sx[j], sy[j]);
                continue;
            }
            else if (isNaN(sx[j]) || isNaN(sy[j])) {
                ctx.stroke();
                ctx.beginPath();
                continue;
            }
            else
                ctx.lineTo(sx[j], sy[j]);
        }
        ctx.stroke();
    }
}
SplineView.__name__ = "SplineView";
export class Spline extends XYGlyph {
    constructor(attrs) {
        super(attrs);
    }
    static init_Spline() {
        this.prototype.default_view = SplineView;
        this.mixins(mixins.LineScalar);
        this.define(({ Boolean, Number }) => ({
            tension: [Number, 0.5],
            closed: [Boolean, false],
        }));
    }
}
Spline.__name__ = "Spline";
Spline.init_Spline();
//# sourceMappingURL=spline.js.map