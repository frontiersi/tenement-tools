import { ActionTool, ActionToolView } from "./action_tool";
import { Dimensions } from "../../../core/enums";
import { scale_range } from "../../../core/util/zoom";
export class ZoomBaseToolView extends ActionToolView {
    doit() {
        const frame = this.plot_view.frame;
        const dims = this.model.dimensions;
        // restrict to axis configured in tool's dimensions property
        const h_axis = dims == 'width' || dims == 'both';
        const v_axis = dims == 'height' || dims == 'both';
        const zoom_info = scale_range(frame, this.model.sign * this.model.factor, h_axis, v_axis);
        this.plot_view.state.push("zoom_out", { range: zoom_info });
        this.plot_view.update_range(zoom_info, { scrolling: true });
        this.model.document?.interactive_start(this.plot_model);
    }
}
ZoomBaseToolView.__name__ = "ZoomBaseToolView";
export class ZoomBaseTool extends ActionTool {
    constructor(attrs) {
        super(attrs);
    }
    static init_ZoomBaseTool() {
        this.define(({ Percent }) => ({
            factor: [Percent, 0.1],
            dimensions: [Dimensions, "both"],
        }));
    }
    get tooltip() {
        return this._get_dim_tooltip(this.dimensions);
    }
}
ZoomBaseTool.__name__ = "ZoomBaseTool";
ZoomBaseTool.init_ZoomBaseTool();
//# sourceMappingURL=zoom_base_tool.js.map