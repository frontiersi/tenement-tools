import { ActionTool, ActionToolView } from "./action_tool";
import { tool_icon_redo } from "../../../styles/icons.css";
export class RedoToolView extends ActionToolView {
    connect_signals() {
        super.connect_signals();
        this.connect(this.plot_view.state.changed, () => this.model.disabled = !this.plot_view.state.can_redo);
    }
    doit() {
        this.plot_view.state.redo();
    }
}
RedoToolView.__name__ = "RedoToolView";
export class RedoTool extends ActionTool {
    constructor(attrs) {
        super(attrs);
        this.tool_name = "Redo";
        this.icon = tool_icon_redo;
    }
    static init_RedoTool() {
        this.prototype.default_view = RedoToolView;
        this.override({
            disabled: true,
        });
        this.register_alias("redo", () => new RedoTool());
    }
}
RedoTool.__name__ = "RedoTool";
RedoTool.init_RedoTool();
//# sourceMappingURL=redo_tool.js.map