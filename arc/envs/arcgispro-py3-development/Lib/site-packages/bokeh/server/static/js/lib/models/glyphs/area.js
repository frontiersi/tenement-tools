import { Glyph, GlyphView } from "./glyph";
import { generic_area_scalar_legend } from "./utils";
import * as mixins from "../../core/property_mixins";
export class AreaView extends GlyphView {
    draw_legend_for_index(ctx, bbox, _index) {
        generic_area_scalar_legend(this.visuals, ctx, bbox);
    }
}
AreaView.__name__ = "AreaView";
export class Area extends Glyph {
    constructor(attrs) {
        super(attrs);
    }
    static init_Area() {
        this.mixins([mixins.FillScalar, mixins.HatchScalar]);
    }
}
Area.__name__ = "Area";
Area.init_Area();
//# sourceMappingURL=area.js.map