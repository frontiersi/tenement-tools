import { ActionTool, ActionToolView } from "./action_tool";
import { tool_icon_help } from "../../../styles/icons.css";
export class HelpToolView extends ActionToolView {
    doit() {
        window.open(this.model.redirect);
    }
}
HelpToolView.__name__ = "HelpToolView";
export class HelpTool extends ActionTool {
    constructor(attrs) {
        super(attrs);
        this.tool_name = "Help";
        this.icon = tool_icon_help;
    }
    static init_HelpTool() {
        this.prototype.default_view = HelpToolView;
        this.define(({ String }) => ({
            redirect: [String, "https://docs.bokeh.org/en/latest/docs/user_guide/tools.html"],
        }));
        this.override({
            description: "Click the question mark to learn more about Bokeh plot tools.",
        });
        this.register_alias("help", () => new HelpTool());
    }
}
HelpTool.__name__ = "HelpTool";
HelpTool.init_HelpTool();
//# sourceMappingURL=help_tool.js.map