import { Annotation, AnnotationView } from "./annotation";
import * as mixins from "../../core/property_mixins";
import { SpatialUnits, RenderMode } from "../../core/enums";
import { BBox } from "../../core/util/bbox";
export const EDGE_TOLERANCE = 2.5;
export class BoxAnnotationView extends AnnotationView {
    constructor() {
        super(...arguments);
        this.bbox = new BBox();
    }
    connect_signals() {
        super.connect_signals();
        this.connect(this.model.change, () => this.request_render());
    }
    _render() {
        const { left, right, top, bottom } = this.model;
        // don't render if *all* position are null
        if (left == null && right == null && top == null && bottom == null)
            return;
        const { frame } = this.plot_view;
        const xscale = this.coordinates.x_scale;
        const yscale = this.coordinates.y_scale;
        const _calc_dim = (dim, dim_units, scale, view, frame_extrema) => {
            let sdim;
            if (dim != null) {
                if (this.model.screen)
                    sdim = dim;
                else {
                    if (dim_units == 'data')
                        sdim = scale.compute(dim);
                    else
                        sdim = view.compute(dim);
                }
            }
            else
                sdim = frame_extrema;
            return sdim;
        };
        this.bbox = BBox.from_rect({
            left: _calc_dim(left, this.model.left_units, xscale, frame.bbox.xview, frame.bbox.left),
            right: _calc_dim(right, this.model.right_units, xscale, frame.bbox.xview, frame.bbox.right),
            top: _calc_dim(top, this.model.top_units, yscale, frame.bbox.yview, frame.bbox.top),
            bottom: _calc_dim(bottom, this.model.bottom_units, yscale, frame.bbox.yview, frame.bbox.bottom),
        });
        this._paint_box();
    }
    _paint_box() {
        const { ctx } = this.layer;
        ctx.save();
        const { left, top, width, height } = this.bbox;
        ctx.beginPath();
        ctx.rect(left, top, width, height);
        if (this.visuals.fill.doit) {
            this.visuals.fill.set_value(ctx);
            ctx.fill();
        }
        if (this.visuals.hatch.doit) {
            this.visuals.hatch.set_value(ctx);
            ctx.fill();
        }
        if (this.visuals.line.doit) {
            this.visuals.line.set_value(ctx);
            ctx.stroke();
        }
        ctx.restore();
    }
    interactive_bbox() {
        const tolerance = this.model.line_width + EDGE_TOLERANCE;
        return this.bbox.grow_by(tolerance);
    }
    interactive_hit(sx, sy) {
        if (this.model.in_cursor == null)
            return false;
        const bbox = this.interactive_bbox();
        return bbox.contains(sx, sy);
    }
    cursor(sx, sy) {
        const tol = 3;
        const { left, right, bottom, top } = this.bbox;
        if (Math.abs(sx - left) < tol || Math.abs(sx - right) < tol)
            return this.model.ew_cursor;
        else if (Math.abs(sy - bottom) < tol || Math.abs(sy - top) < tol)
            return this.model.ns_cursor;
        else if (this.bbox.contains(sx, sy))
            return this.model.in_cursor;
        else
            return null;
    }
}
BoxAnnotationView.__name__ = "BoxAnnotationView";
export class BoxAnnotation extends Annotation {
    constructor(attrs) {
        super(attrs);
    }
    static init_BoxAnnotation() {
        this.prototype.default_view = BoxAnnotationView;
        this.mixins([mixins.Line, mixins.Fill, mixins.Hatch]);
        this.define(({ Number, Nullable }) => ({
            top: [Nullable(Number), null],
            top_units: [SpatialUnits, "data"],
            bottom: [Nullable(Number), null],
            bottom_units: [SpatialUnits, "data"],
            left: [Nullable(Number), null],
            left_units: [SpatialUnits, "data"],
            right: [Nullable(Number), null],
            right_units: [SpatialUnits, "data"],
            /** @deprecated */
            render_mode: [RenderMode, "canvas"],
        }));
        this.internal(({ Boolean, String, Nullable }) => ({
            screen: [Boolean, false],
            ew_cursor: [Nullable(String), null],
            ns_cursor: [Nullable(String), null],
            in_cursor: [Nullable(String), null],
        }));
        this.override({
            fill_color: '#fff9ba',
            fill_alpha: 0.4,
            line_color: '#cccccc',
            line_alpha: 0.3,
        });
    }
    update({ left, right, top, bottom }) {
        this.setv({ left, right, top, bottom, screen: true });
    }
}
BoxAnnotation.__name__ = "BoxAnnotation";
BoxAnnotation.init_BoxAnnotation();
//# sourceMappingURL=box_annotation.js.map