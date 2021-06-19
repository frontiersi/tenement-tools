import { Texture } from "./texture";
import { ImageLoader } from "../../core/util/image";
export class ImageURLTexture extends Texture {
    constructor(attrs) {
        super(attrs);
    }
    static init_ImageURLTexture() {
        this.define(({ String }) => ({
            url: [String],
        }));
    }
    initialize() {
        super.initialize();
        this._loader = new ImageLoader(this.url);
    }
    get_pattern(_color, _scale, _weight) {
        const { _loader } = this;
        return this._loader.finished ? _loader.image : _loader.promise;
    }
}
ImageURLTexture.__name__ = "ImageURLTexture";
ImageURLTexture.init_ImageURLTexture();
//# sourceMappingURL=image_url_texture.js.map