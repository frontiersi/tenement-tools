import { VisualProperties, VisualUniforms } from "./visual";
import * as mixins from "../property_mixins";
import { color2css } from "../util/color";
export class Fill extends VisualProperties {
    get doit() {
        const color = this.fill_color.get_value();
        const alpha = this.fill_alpha.get_value();
        return !(color == null || alpha == 0);
    }
    set_value(ctx) {
        const color = this.fill_color.get_value();
        const alpha = this.fill_alpha.get_value();
        ctx.fillStyle = color2css(color, alpha);
    }
}
Fill.__name__ = "Fill";
export class FillScalar extends VisualUniforms {
    get doit() {
        const color = this.fill_color.value;
        const alpha = this.fill_alpha.value;
        return !(color == 0 || alpha == 0);
    }
    set_value(ctx) {
        const color = this.fill_color.value;
        const alpha = this.fill_alpha.value;
        ctx.fillStyle = color2css(color, alpha);
    }
}
FillScalar.__name__ = "FillScalar";
export class FillVector extends VisualUniforms {
    get doit() {
        const { fill_color } = this;
        if (fill_color.is_Scalar() && fill_color.value == 0)
            return false;
        const { fill_alpha } = this;
        if (fill_alpha.is_Scalar() && fill_alpha.value == 0)
            return false;
        return true;
    }
    set_vectorize(ctx, i) {
        const color = this.fill_color.get(i);
        const alpha = this.fill_alpha.get(i);
        ctx.fillStyle = color2css(color, alpha);
    }
}
FillVector.__name__ = "FillVector";
Fill.prototype.type = "fill";
Fill.prototype.attrs = Object.keys(mixins.Fill);
FillScalar.prototype.type = "fill";
FillScalar.prototype.attrs = Object.keys(mixins.FillScalar);
FillVector.prototype.type = "fill";
FillVector.prototype.attrs = Object.keys(mixins.FillVector);
//# sourceMappingURL=fill.js.map