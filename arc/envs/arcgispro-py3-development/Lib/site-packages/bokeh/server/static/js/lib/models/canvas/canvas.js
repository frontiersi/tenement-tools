import { HasProps } from "../../core/has_props";
import { DOMView } from "../../core/dom_view";
import { logger } from "../../core/logging";
import { div, append } from "../../core/dom";
import { OutputBackend } from "../../core/enums";
import { extend } from "../../core/util/object";
import { UIEventBus } from "../../core/ui_events";
import { BBox } from "../../core/util/bbox";
import { CanvasLayer } from "../../core/util/canvas";
const global_webgl = (() => {
    // We use a global invisible canvas and gl context. By having a global context,
    // we avoid the limitation of max 16 contexts that most browsers have.
    const canvas = document.createElement("canvas");
    const gl = canvas.getContext("webgl", { premultipliedAlpha: true });
    // If WebGL is available, we store a reference to the gl canvas on
    // the ctx object, because that's what gets passed everywhere.
    if (gl != null)
        return { canvas, gl };
    else {
        logger.trace("WebGL is not supported");
        return undefined;
    }
})();
const style = {
    position: "absolute",
    top: "0",
    left: "0",
    width: "100%",
    height: "100%",
};
export class CanvasView extends DOMView {
    constructor() {
        super(...arguments);
        this.bbox = new BBox();
    }
    initialize() {
        super.initialize();
        if (this.model.output_backend == "webgl") {
            this.webgl = global_webgl;
        }
        this.underlays_el = div({ style });
        this.primary = this.create_layer();
        this.overlays = this.create_layer();
        this.overlays_el = div({ style });
        this.events_el = div({ class: "bk-canvas-events", style });
        const elements = [
            this.underlays_el,
            this.primary.el,
            this.overlays.el,
            this.overlays_el,
            this.events_el,
        ];
        extend(this.el.style, style);
        append(this.el, ...elements);
        this.ui_event_bus = new UIEventBus(this);
    }
    remove() {
        this.ui_event_bus.destroy();
        super.remove();
    }
    add_underlay(el) {
        this.underlays_el.appendChild(el);
    }
    add_overlay(el) {
        this.overlays_el.appendChild(el);
    }
    add_event(el) {
        this.events_el.appendChild(el);
    }
    get pixel_ratio() {
        return this.primary.pixel_ratio; // XXX: primary
    }
    resize(width, height) {
        this.bbox = new BBox({ left: 0, top: 0, width, height });
        this.primary.resize(width, height);
        this.overlays.resize(width, height);
    }
    prepare_webgl(frame_box) {
        // Prepare WebGL for a drawing pass
        const { webgl } = this;
        if (webgl != null) {
            // Sync canvas size
            const { width, height } = this.bbox;
            webgl.canvas.width = this.pixel_ratio * width;
            webgl.canvas.height = this.pixel_ratio * height;
            const { gl } = webgl;
            // Clipping
            gl.enable(gl.SCISSOR_TEST);
            const [sx, sy, w, h] = frame_box;
            const { xview, yview } = this.bbox;
            const vx = xview.compute(sx);
            const vy = yview.compute(sy + h);
            const ratio = this.pixel_ratio;
            gl.scissor(ratio * vx, ratio * vy, ratio * w, ratio * h); // lower left corner, width, height
            // Setup blending
            gl.enable(gl.BLEND);
            gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE_MINUS_DST_ALPHA, gl.ONE); // premultipliedAlpha == true
            this._clear_webgl();
        }
    }
    blit_webgl(ctx) {
        // This should be called when the ctx has no state except the HIDPI transform
        const { webgl } = this;
        if (webgl != null) {
            // Blit gl canvas into the 2D canvas. To do 1-on-1 blitting, we need
            // to remove the hidpi transform, then blit, then restore.
            // ctx.globalCompositeOperation = "source-over"  -> OK; is the default
            logger.debug('Blitting WebGL canvas');
            ctx.restore();
            ctx.drawImage(webgl.canvas, 0, 0);
            // Set back hidpi transform
            ctx.save();
            if (this.model.hidpi) {
                const ratio = this.pixel_ratio;
                ctx.scale(ratio, ratio);
                ctx.translate(0.5, 0.5);
            }
            this._clear_webgl();
        }
    }
    _clear_webgl() {
        const { webgl } = this;
        if (webgl != null) {
            // Prepare GL for drawing
            const { gl, canvas } = webgl;
            gl.viewport(0, 0, canvas.width, canvas.height);
            gl.clearColor(0, 0, 0, 0);
            gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        }
    }
    compose() {
        const composite = this.create_layer();
        const { width, height } = this.bbox;
        composite.resize(width, height);
        composite.ctx.drawImage(this.primary.canvas, 0, 0);
        composite.ctx.drawImage(this.overlays.canvas, 0, 0);
        return composite;
    }
    create_layer() {
        const { output_backend, hidpi } = this.model;
        return new CanvasLayer(output_backend, hidpi);
    }
    to_blob() {
        return this.compose().to_blob();
    }
}
CanvasView.__name__ = "CanvasView";
export class Canvas extends HasProps {
    constructor(attrs) {
        super(attrs);
    }
    static init_Canvas() {
        this.prototype.default_view = CanvasView;
        this.internal(({ Boolean }) => ({
            hidpi: [Boolean, true],
            output_backend: [OutputBackend, "canvas"],
        }));
    }
}
Canvas.__name__ = "Canvas";
Canvas.init_Canvas();
//# sourceMappingURL=canvas.js.map