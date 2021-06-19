import { TickFormatter } from "./tick_formatter";
import { BasicTickFormatter, unicode_replace } from "./basic_tick_formatter";
import { LogTicker } from "../tickers/log_ticker";
import { BaseExpo, TextBox } from "../../core/graphics";
const { log, round } = Math;
export class LogTickFormatter extends TickFormatter {
    constructor(attrs) {
        super(attrs);
    }
    static init_LogTickFormatter() {
        this.define(({ Ref, Nullable }) => ({
            ticker: [Nullable(Ref(LogTicker)), null],
        }));
    }
    initialize() {
        super.initialize();
        this.basic_formatter = new BasicTickFormatter();
    }
    format_graphics(ticks, opts) {
        if (ticks.length == 0)
            return [];
        const base = this.ticker?.base ?? 10;
        const expos = this._exponents(ticks, base);
        if (expos == null)
            return this.basic_formatter.format_graphics(ticks, opts);
        else {
            return expos.map((expo) => {
                const b = new TextBox({ text: unicode_replace(`${base}`) });
                const e = new TextBox({ text: unicode_replace(`${expo}`) });
                return new BaseExpo(b, e);
            });
        }
    }
    _exponents(ticks, base) {
        let last_exponent = null;
        const exponents = [];
        for (const tick of ticks) {
            const exponent = round(log(tick) / log(base));
            if (last_exponent != exponent) {
                last_exponent = exponent;
                exponents.push(exponent);
            }
            else
                return null;
        }
        return exponents;
    }
    doFormat(ticks, opts) {
        if (ticks.length == 0)
            return [];
        const base = this.ticker?.base ?? 10;
        const expos = this._exponents(ticks, base);
        if (expos == null)
            return this.basic_formatter.doFormat(ticks, opts);
        else
            return expos.map((expo) => unicode_replace(`${base}^${expo}`));
    }
}
LogTickFormatter.__name__ = "LogTickFormatter";
LogTickFormatter.init_LogTickFormatter();
//# sourceMappingURL=log_tick_formatter.js.map