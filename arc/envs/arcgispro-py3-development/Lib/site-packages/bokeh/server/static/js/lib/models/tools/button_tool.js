import Hammer from "hammerjs";
import { DOMView } from "../../core/dom_view";
import { Tool, ToolView } from "./tool";
import { empty } from "../../core/dom";
import { startsWith } from "../../core/util/string";
import { isString } from "../../core/util/types";
import { reversed } from "../../core/util/array";
import toolbar_css, * as toolbars from "../../styles/toolbar.css";
import icons_css from "../../styles/icons.css";
import menus_css from "../../styles/menus.css";
import { ContextMenu } from "../../core/util/menus";
export class ButtonToolButtonView extends DOMView {
    initialize() {
        super.initialize();
        const items = this.model.menu;
        if (items != null) {
            const location = this.parent.model.toolbar_location;
            const reverse = location == "left" || location == "above";
            const orientation = this.parent.model.horizontal ? "vertical" : "horizontal";
            this._menu = new ContextMenu(!reverse ? items : reversed(items), {
                orientation,
                prevent_hide: (event) => event.target == this.el,
            });
        }
        this._hammer = new Hammer(this.el, {
            touchAction: "auto",
            inputClass: Hammer.TouchMouseInput,
        });
        this.connect(this.model.change, () => this.render());
        this._hammer.on("tap", (e) => {
            if (this._menu?.is_open) {
                this._menu.hide();
                return;
            }
            if (e.target == this.el) {
                this._clicked();
            }
        });
        this._hammer.on("press", () => this._pressed());
    }
    remove() {
        this._hammer.destroy();
        this._menu?.remove();
        super.remove();
    }
    styles() {
        return [...super.styles(), toolbar_css, icons_css, menus_css];
    }
    css_classes() {
        return super.css_classes().concat(toolbars.toolbar_button);
    }
    render() {
        empty(this.el);
        const icon = this.model.computed_icon;
        if (isString(icon)) {
            if (startsWith(icon, "data:image"))
                this.el.style.backgroundImage = "url('" + icon + "')";
            else
                this.el.classList.add(icon);
        }
        this.el.title = this.model.tooltip;
        if (this._menu != null) {
            this.root.el.appendChild(this._menu.el);
        }
    }
    _pressed() {
        const { left, top, right, bottom } = this.el.getBoundingClientRect();
        const at = (() => {
            switch (this.parent.model.toolbar_location) {
                case "right":
                    return { right: left, top };
                case "left":
                    return { left: right, top };
                case "above":
                    return { left, top: bottom };
                case "below":
                    return { left, bottom: top };
            }
        })();
        this._menu?.toggle(at);
    }
}
ButtonToolButtonView.__name__ = "ButtonToolButtonView";
export class ButtonToolView extends ToolView {
}
ButtonToolView.__name__ = "ButtonToolView";
export class ButtonTool extends Tool {
    constructor(attrs) {
        super(attrs);
    }
    static init_ButtonTool() {
        this.internal(({ Boolean }) => ({
            disabled: [Boolean, false],
        }));
    }
    // utility function to return a tool name, modified
    // by the active dimensions. Used by tools that have dimensions
    _get_dim_tooltip(dims) {
        const { description, tool_name } = this;
        if (description != null)
            return description;
        else if (dims == "both")
            return tool_name;
        else
            return `${tool_name} (${dims == "width" ? "x" : "y"}-axis)`;
    }
    get tooltip() {
        return this.description ?? this.tool_name;
    }
    get computed_icon() {
        return this.icon;
    }
    get menu() {
        return null;
    }
}
ButtonTool.__name__ = "ButtonTool";
ButtonTool.init_ButtonTool();
//# sourceMappingURL=button_tool.js.map