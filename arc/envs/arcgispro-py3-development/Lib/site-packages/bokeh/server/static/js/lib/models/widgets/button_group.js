import { Control, ControlView } from "./control";
import { ButtonType } from "../../core/enums";
import { div } from "../../core/dom";
import buttons_css, * as buttons from "../../styles/buttons.css";
export class ButtonGroupView extends ControlView {
    *controls() {
        yield* this._buttons; // TODO: HTMLButtonElement[]
    }
    connect_signals() {
        super.connect_signals();
        const p = this.model.properties;
        this.on_change(p.button_type, () => this.render());
        this.on_change(p.labels, () => this.render());
        this.on_change(p.active, () => this._update_active());
    }
    styles() {
        return [...super.styles(), buttons_css];
    }
    render() {
        super.render();
        this._buttons = this.model.labels.map((label, i) => {
            const button = div({
                class: [buttons.btn, buttons[`btn_${this.model.button_type}`]],
                disabled: this.model.disabled,
            }, label);
            button.addEventListener("click", () => this.change_active(i));
            return button;
        });
        this._update_active();
        const group = div({ class: buttons.btn_group }, this._buttons);
        this.el.appendChild(group);
    }
}
ButtonGroupView.__name__ = "ButtonGroupView";
export class ButtonGroup extends Control {
    constructor(attrs) {
        super(attrs);
    }
    static init_ButtonGroup() {
        this.define(({ String, Array }) => ({
            labels: [Array(String), []],
            button_type: [ButtonType, "default"],
        }));
    }
}
ButtonGroup.__name__ = "ButtonGroup";
ButtonGroup.init_ButtonGroup();
//# sourceMappingURL=button_group.js.map