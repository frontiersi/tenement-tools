import { ActionTool, ActionToolView, ActionToolButtonView } from "./action_tool";
export class CustomActionButtonView extends ActionToolButtonView {
    css_classes() {
        return super.css_classes().concat("bk-toolbar-button-custom-action");
    }
}
CustomActionButtonView.__name__ = "CustomActionButtonView";
export class CustomActionView extends ActionToolView {
    doit() {
        this.model.callback?.execute(this.model);
    }
}
CustomActionView.__name__ = "CustomActionView";
export class CustomAction extends ActionTool {
    constructor(attrs) {
        super(attrs);
        this.tool_name = "Custom Action";
        this.button_view = CustomActionButtonView;
    }
    static init_CustomAction() {
        this.prototype.default_view = CustomActionView;
        this.define(({ Any, String, Nullable }) => ({
            callback: [Nullable(Any /*TODO*/)],
            icon: [String],
        }));
        this.override({
            description: "Perform a Custom Action",
        });
    }
}
CustomAction.__name__ = "CustomAction";
CustomAction.init_CustomAction();
//# sourceMappingURL=custom_action.js.map