import { Model } from "../../model";
import { View } from "../../core/view";
import * as visuals from "../../core/visuals";
import { LineVector, FillVector } from "../../core/property_mixins";
import * as p from "../../core/properties";
export class ArrowHeadView extends View {
    initialize() {
        super.initialize();
        this.visuals = new visuals.Visuals(this);
    }
    request_render() {
        this.parent.request_render();
    }
    get canvas() {
        return this.parent.canvas;
    }
    set_data(source) {
        const self = this;
        for (const prop of this.model) {
            if (!(prop instanceof p.VectorSpec || prop instanceof p.ScalarSpec))
                continue;
            const uniform = prop.uniform(source);
            self[`${prop.attr}`] = uniform;
        }
    }
}
ArrowHeadView.__name__ = "ArrowHeadView";
export class ArrowHead extends Model {
    constructor(attrs) {
        super(attrs);
    }
    static init_ArrowHead() {
        this.define(() => ({
            size: [p.NumberSpec, 25],
        }));
    }
}
ArrowHead.__name__ = "ArrowHead";
ArrowHead.init_ArrowHead();
export class OpenHeadView extends ArrowHeadView {
    clip(ctx, i) {
        this.visuals.line.set_vectorize(ctx, i);
        const size_i = this.size.get(i);
        ctx.moveTo(0.5 * size_i, size_i);
        ctx.lineTo(0.5 * size_i, -2);
        ctx.lineTo(-0.5 * size_i, -2);
        ctx.lineTo(-0.5 * size_i, size_i);
        ctx.lineTo(0, 0);
        ctx.lineTo(0.5 * size_i, size_i);
    }
    render(ctx, i) {
        if (this.visuals.line.doit) {
            this.visuals.line.set_vectorize(ctx, i);
            const size_i = this.size.get(i);
            ctx.beginPath();
            ctx.moveTo(0.5 * size_i, size_i);
            ctx.lineTo(0, 0);
            ctx.lineTo(-0.5 * size_i, size_i);
            ctx.stroke();
        }
    }
}
OpenHeadView.__name__ = "OpenHeadView";
export class OpenHead extends ArrowHead {
    constructor(attrs) {
        super(attrs);
    }
    static init_OpenHead() {
        this.prototype.default_view = OpenHeadView;
        this.mixins(LineVector);
    }
}
OpenHead.__name__ = "OpenHead";
OpenHead.init_OpenHead();
export class NormalHeadView extends ArrowHeadView {
    clip(ctx, i) {
        this.visuals.line.set_vectorize(ctx, i);
        const size_i = this.size.get(i);
        ctx.moveTo(0.5 * size_i, size_i);
        ctx.lineTo(0.5 * size_i, -2);
        ctx.lineTo(-0.5 * size_i, -2);
        ctx.lineTo(-0.5 * size_i, size_i);
        ctx.lineTo(0.5 * size_i, size_i);
    }
    render(ctx, i) {
        if (this.visuals.fill.doit) {
            this.visuals.fill.set_vectorize(ctx, i);
            this._normal(ctx, i);
            ctx.fill();
        }
        if (this.visuals.line.doit) {
            this.visuals.line.set_vectorize(ctx, i);
            this._normal(ctx, i);
            ctx.stroke();
        }
    }
    _normal(ctx, i) {
        const size_i = this.size.get(i);
        ctx.beginPath();
        ctx.moveTo(0.5 * size_i, size_i);
        ctx.lineTo(0, 0);
        ctx.lineTo(-0.5 * size_i, size_i);
        ctx.closePath();
    }
}
NormalHeadView.__name__ = "NormalHeadView";
export class NormalHead extends ArrowHead {
    constructor(attrs) {
        super(attrs);
    }
    static init_NormalHead() {
        this.prototype.default_view = NormalHeadView;
        this.mixins([LineVector, FillVector]);
        this.override({
            fill_color: "black",
        });
    }
}
NormalHead.__name__ = "NormalHead";
NormalHead.init_NormalHead();
export class VeeHeadView extends ArrowHeadView {
    clip(ctx, i) {
        this.visuals.line.set_vectorize(ctx, i);
        const size_i = this.size.get(i);
        ctx.moveTo(0.5 * size_i, size_i);
        ctx.lineTo(0.5 * size_i, -2);
        ctx.lineTo(-0.5 * size_i, -2);
        ctx.lineTo(-0.5 * size_i, size_i);
        ctx.lineTo(0, 0.5 * size_i);
        ctx.lineTo(0.5 * size_i, size_i);
    }
    render(ctx, i) {
        if (this.visuals.fill.doit) {
            this.visuals.fill.set_vectorize(ctx, i);
            this._vee(ctx, i);
            ctx.fill();
        }
        if (this.visuals.line.doit) {
            this.visuals.line.set_vectorize(ctx, i);
            this._vee(ctx, i);
            ctx.stroke();
        }
    }
    _vee(ctx, i) {
        const size_i = this.size.get(i);
        ctx.beginPath();
        ctx.moveTo(0.5 * size_i, size_i);
        ctx.lineTo(0, 0);
        ctx.lineTo(-0.5 * size_i, size_i);
        ctx.lineTo(0, 0.5 * size_i);
        ctx.closePath();
    }
}
VeeHeadView.__name__ = "VeeHeadView";
export class VeeHead extends ArrowHead {
    constructor(attrs) {
        super(attrs);
    }
    static init_VeeHead() {
        this.prototype.default_view = VeeHeadView;
        this.mixins([LineVector, FillVector]);
        this.override({
            fill_color: "black",
        });
    }
}
VeeHead.__name__ = "VeeHead";
VeeHead.init_VeeHead();
export class TeeHeadView extends ArrowHeadView {
    render(ctx, i) {
        if (this.visuals.line.doit) {
            this.visuals.line.set_vectorize(ctx, i);
            const size_i = this.size.get(i);
            ctx.beginPath();
            ctx.moveTo(0.5 * size_i, 0);
            ctx.lineTo(-0.5 * size_i, 0);
            ctx.stroke();
        }
    }
    clip(_ctx, _i) { }
}
TeeHeadView.__name__ = "TeeHeadView";
export class TeeHead extends ArrowHead {
    constructor(attrs) {
        super(attrs);
    }
    static init_TeeHead() {
        this.prototype.default_view = TeeHeadView;
        this.mixins(LineVector);
    }
}
TeeHead.__name__ = "TeeHead";
TeeHead.init_TeeHead();
//# sourceMappingURL=arrow_head.js.map