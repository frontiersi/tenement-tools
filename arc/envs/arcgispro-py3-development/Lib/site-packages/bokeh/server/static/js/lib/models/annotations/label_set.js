import { TextAnnotation, TextAnnotationView } from "./text_annotation";
import { DataAnnotationView } from "./data_annotation";
import { ColumnDataSource } from "../sources/column_data_source";
import * as mixins from "../../core/property_mixins";
import { SpatialUnits } from "../../core/enums";
import { div, display } from "../../core/dom";
import * as p from "../../core/properties";
import { font_metrics } from "../../core/util/text";
export class LabelSetView extends TextAnnotationView {
    // XXX: can't inherit DataAnnotation currently
    set_data(source) {
        DataAnnotationView.prototype.set_data.call(this, source);
    }
    initialize() {
        super.initialize();
        this.set_data(this.model.source);
        if (this.model.render_mode == 'css') {
            for (let i = 0, end = this.text.length; i < end; i++) {
                const el = div({ style: { display: "none" } });
                this.el.appendChild(el);
            }
        }
    }
    connect_signals() {
        super.connect_signals();
        const render = () => {
            this.set_data(this.model.source);
            if (this.model.render_mode == "css")
                this.render();
            else
                this.request_render();
        };
        this.connect(this.model.change, render);
        this.connect(this.model.source.streaming, render);
        this.connect(this.model.source.patching, render);
        this.connect(this.model.source.change, render);
    }
    _calculate_text_dimensions(ctx, text) {
        const { width } = ctx.measureText(text);
        const { height } = font_metrics(this.visuals.text.font_value(0));
        return [width, height];
    }
    _map_data() {
        const xscale = this.coordinates.x_scale;
        const yscale = this.coordinates.y_scale;
        const panel = this.layout != null ? this.layout : this.plot_view.frame;
        const sx = this.model.x_units == "data" ? xscale.v_compute(this._x) : panel.bbox.xview.v_compute(this._x);
        const sy = this.model.y_units == "data" ? yscale.v_compute(this._y) : panel.bbox.yview.v_compute(this._y);
        return [sx, sy];
    }
    _render() {
        const draw = this.model.render_mode == 'canvas' ? this._v_canvas_text.bind(this) : this._v_css_text.bind(this);
        const { ctx } = this.layer;
        const [sx, sy] = this._map_data();
        for (let i = 0, end = this.text.length; i < end; i++) {
            draw(ctx, i, this.text.get(i), sx[i] + this.x_offset.get(i), sy[i] - this.y_offset.get(i), this.angle.get(i));
        }
    }
    _get_size() {
        const { ctx } = this.layer;
        this.visuals.text.set_vectorize(ctx, 0);
        const { width } = ctx.measureText(this.text.get(0));
        const { height } = font_metrics(ctx.font);
        return { width, height };
    }
    _v_canvas_text(ctx, i, text, sx, sy, angle) {
        this.visuals.text.set_vectorize(ctx, i);
        const bbox_dims = this._calculate_bounding_box_dimensions(ctx, text);
        ctx.save();
        ctx.beginPath();
        ctx.translate(sx, sy);
        ctx.rotate(angle);
        ctx.rect(bbox_dims[0], bbox_dims[1], bbox_dims[2], bbox_dims[3]);
        if (this.visuals.background_fill.doit) {
            this.visuals.background_fill.set_vectorize(ctx, i);
            ctx.fill();
        }
        if (this.visuals.border_line.doit) {
            this.visuals.border_line.set_vectorize(ctx, i);
            ctx.stroke();
        }
        if (this.visuals.text.doit) {
            this.visuals.text.set_vectorize(ctx, i);
            ctx.fillText(text, 0, 0);
        }
        ctx.restore();
    }
    _v_css_text(ctx, i, text, sx, sy, angle) {
        const el = this.el.children[i];
        el.textContent = text;
        this.visuals.text.set_vectorize(ctx, i);
        const [x, y] = this._calculate_bounding_box_dimensions(ctx, text);
        el.style.position = "absolute";
        el.style.left = `${sx + x}px`;
        el.style.top = `${sy + y}px`;
        el.style.color = ctx.fillStyle;
        el.style.font = ctx.font;
        el.style.lineHeight = "normal"; // needed to prevent ipynb css override
        if (angle) {
            el.style.transform = `rotate(${angle}rad)`;
        }
        if (this.visuals.background_fill.doit) {
            this.visuals.background_fill.set_vectorize(ctx, i);
            el.style.backgroundColor = ctx.fillStyle;
        }
        if (this.visuals.border_line.doit) {
            this.visuals.border_line.set_vectorize(ctx, i);
            // attempt to support vector-style ("8 4 8") line dashing for css mode
            el.style.borderStyle = ctx.lineDash.length < 2 ? "solid" : "dashed";
            el.style.borderWidth = `${ctx.lineWidth}px`;
            el.style.borderColor = ctx.strokeStyle;
        }
        display(el);
    }
}
LabelSetView.__name__ = "LabelSetView";
export class LabelSet extends TextAnnotation {
    constructor(attrs) {
        super(attrs);
    }
    static init_LabelSet() {
        this.prototype.default_view = LabelSetView;
        this.mixins([
            mixins.TextVector,
            ["border_", mixins.LineVector],
            ["background_", mixins.FillVector],
        ]);
        this.define(({ Ref }) => ({
            x: [p.XCoordinateSpec, { field: "x" }],
            y: [p.YCoordinateSpec, { field: "y" }],
            x_units: [SpatialUnits, "data"],
            y_units: [SpatialUnits, "data"],
            text: [p.StringSpec, { field: "text" }],
            angle: [p.AngleSpec, 0],
            x_offset: [p.NumberSpec, { value: 0 }],
            y_offset: [p.NumberSpec, { value: 0 }],
            source: [Ref(ColumnDataSource), () => new ColumnDataSource()],
        }));
        this.override({
            background_fill_color: null,
            border_line_color: null,
        });
    }
}
LabelSet.__name__ = "LabelSet";
LabelSet.init_LabelSet();
//# sourceMappingURL=label_set.js.map