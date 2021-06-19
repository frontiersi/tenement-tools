import { TextLikeInput, TextLikeInputView } from "./text_like_input";
import { textarea } from "../../core/dom";
import * as inputs from "../../styles/widgets/inputs.css";
export class TextAreaInputView extends TextLikeInputView {
    connect_signals() {
        super.connect_signals();
        this.connect(this.model.properties.rows.change, () => this.input_el.rows = this.model.rows);
        this.connect(this.model.properties.cols.change, () => this.input_el.cols = this.model.cols);
    }
    _render_input() {
        this.input_el = textarea({ class: inputs.input });
    }
    render() {
        super.render();
        this.input_el.cols = this.model.cols;
        this.input_el.rows = this.model.rows;
    }
}
TextAreaInputView.__name__ = "TextAreaInputView";
export class TextAreaInput extends TextLikeInput {
    constructor(attrs) {
        super(attrs);
    }
    static init_TextAreaInput() {
        this.prototype.default_view = TextAreaInputView;
        this.define(({ Int }) => ({
            cols: [Int, 20],
            rows: [Int, 2],
        }));
        this.override({
            max_length: 500,
        });
    }
}
TextAreaInput.__name__ = "TextAreaInput";
TextAreaInput.init_TextAreaInput();
//# sourceMappingURL=textarea_input.js.map