import { EllipseOval, EllipseOvalView } from "./ellipse_oval";
import { mul } from "../../core/util/arrayable";
export class OvalView extends EllipseOvalView {
    _map_data() {
        super._map_data();
        mul(this.sw, 0.75); // oval drawn from bezier curves = ellipse with width reduced by 3/4
    }
}
OvalView.__name__ = "OvalView";
export class Oval extends EllipseOval {
    constructor(attrs) {
        super(attrs);
    }
    static init_Oval() {
        this.prototype.default_view = OvalView;
    }
}
Oval.__name__ = "Oval";
Oval.init_Oval();
//# sourceMappingURL=oval.js.map