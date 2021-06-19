import { Model } from "../../model";
import { LayoutDOM } from "./layout_dom";
export class Panel extends Model {
    constructor(attrs) {
        super(attrs);
    }
    static init_Panel() {
        this.define(({ Boolean, String, Ref }) => ({
            title: [String, ""],
            child: [Ref(LayoutDOM)],
            closable: [Boolean, false],
        }));
    }
}
Panel.__name__ = "Panel";
Panel.init_Panel();
//# sourceMappingURL=panel.js.map