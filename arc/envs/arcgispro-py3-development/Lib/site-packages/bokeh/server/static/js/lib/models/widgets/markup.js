import { CachedVariadicBox } from "../../core/layout/html";
import { div } from "../../core/dom";
import { Widget, WidgetView } from "./widget";
import clearfix_css, { clearfix } from "../../styles/clearfix.css";
export class MarkupView extends WidgetView {
    connect_signals() {
        super.connect_signals();
        this.connect(this.model.change, () => {
            this.layout.invalidate_cache();
            this.render();
            this.root.compute_layout(); // XXX: invalidate_layout?
        });
    }
    styles() {
        return [...super.styles(), clearfix_css];
    }
    _update_layout() {
        this.layout = new CachedVariadicBox(this.el);
        this.layout.set_sizing(this.box_sizing());
    }
    render() {
        super.render();
        const style = { ...this.model.style, display: "inline-block" };
        this.markup_el = div({ class: clearfix, style });
        this.el.appendChild(this.markup_el);
    }
}
MarkupView.__name__ = "MarkupView";
export class Markup extends Widget {
    constructor(attrs) {
        super(attrs);
    }
    static init_Markup() {
        this.define(({ String, Dict }) => ({
            text: [String, ""],
            style: [Dict(String), {}],
        }));
    }
}
Markup.__name__ = "Markup";
Markup.init_Markup();
//# sourceMappingURL=markup.js.map