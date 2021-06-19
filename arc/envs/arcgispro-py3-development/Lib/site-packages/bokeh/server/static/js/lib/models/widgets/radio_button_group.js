import { ButtonGroup, ButtonGroupView } from "./button_group";
import { classes } from "../../core/dom";
import * as buttons from "../../styles/buttons.css";
export class RadioButtonGroupView extends ButtonGroupView {
    change_active(i) {
        if (this.model.active !== i) {
            this.model.active = i;
        }
    }
    _update_active() {
        const { active } = this.model;
        this._buttons.forEach((button, i) => {
            classes(button).toggle(buttons.active, active === i);
        });
    }
}
RadioButtonGroupView.__name__ = "RadioButtonGroupView";
export class RadioButtonGroup extends ButtonGroup {
    constructor(attrs) {
        super(attrs);
    }
    static init_RadioButtonGroup() {
        this.prototype.default_view = RadioButtonGroupView;
        this.define(({ Int, Nullable }) => ({
            active: [Nullable(Int), null],
        }));
    }
}
RadioButtonGroup.__name__ = "RadioButtonGroup";
RadioButtonGroup.init_RadioButtonGroup();
//# sourceMappingURL=radio_button_group.js.map