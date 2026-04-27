import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type MouseEvent as ReactMouseEvent,
} from "react";
import { Maximize2, RotateCcw, ZoomIn, ZoomOut } from "lucide-react";
import { Button } from "@/components/ui/button";

interface Props {
  image: HTMLImageElement | null;
  isLoading?: boolean;
  error?: string | null;
  overlay?: HTMLImageElement | null;
  overlayOpacity?: number;
}

interface View {
  scale: number;
  panX: number;
  panY: number;
}

const MIN_SCALE = 0.05;
const MAX_SCALE = 20;

export function StudyViewer({
  image,
  isLoading,
  error,
  overlay = null,
  overlayOpacity = 0.5,
}: Props) {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [view, setView] = useState<View>({ scale: 1, panX: 0, panY: 0 });
  const [brightness, setBrightness] = useState(1);
  const [contrast, setContrast] = useState(1);
  const dragRef = useRef<{ x: number; y: number; panX: number; panY: number } | null>(null);

  useEffect(() => {
    const node = wrapperRef.current;
    if (!node) return;
    const ro = new ResizeObserver((entries) => {
      const cr = entries[0].contentRect;
      setSize({ w: Math.floor(cr.width), h: Math.floor(cr.height) });
    });
    ro.observe(node);
    return () => ro.disconnect();
  }, []);

  const fit = useCallback(() => {
    if (!image || size.w === 0 || size.h === 0) return;
    const s = Math.min(size.w / image.naturalWidth, size.h / image.naturalHeight);
    setView({
      scale: s,
      panX: (size.w - image.naturalWidth * s) / 2,
      panY: (size.h - image.naturalHeight * s) / 2,
    });
  }, [image, size]);

  useEffect(() => {
    fit();
  }, [fit]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || size.w === 0 || size.h === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size.w * dpr;
    canvas.height = size.h * dpr;
    canvas.style.width = `${size.w}px`;
    canvas.style.height = `${size.h}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, size.w, size.h);
    if (!image) return;
    ctx.imageSmoothingEnabled = view.scale < 4;
    ctx.filter = `brightness(${brightness}) contrast(${contrast})`;
    const drawW = image.naturalWidth * view.scale;
    const drawH = image.naturalHeight * view.scale;
    ctx.drawImage(image, view.panX, view.panY, drawW, drawH);
    ctx.filter = "none";
    if (overlay) {
      ctx.globalAlpha = Math.max(0, Math.min(1, overlayOpacity));
      ctx.drawImage(overlay, view.panX, view.panY, drawW, drawH);
      ctx.globalAlpha = 1;
    }
  }, [image, size, view, brightness, contrast, overlay, overlayOpacity]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const handler = (e: WheelEvent) => {
      if (!image) return;
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const cx = e.clientX - rect.left;
      const cy = e.clientY - rect.top;
      const factor = Math.exp(-e.deltaY * 0.0015);
      setView((v) => {
        const newScale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, v.scale * factor));
        const ratio = newScale / v.scale;
        return {
          scale: newScale,
          panX: cx - (cx - v.panX) * ratio,
          panY: cy - (cy - v.panY) * ratio,
        };
      });
    };
    canvas.addEventListener("wheel", handler, { passive: false });
    return () => canvas.removeEventListener("wheel", handler);
  }, [image]);

  const onMouseDown = (e: ReactMouseEvent<HTMLCanvasElement>) => {
    if (!image) return;
    dragRef.current = { x: e.clientX, y: e.clientY, panX: view.panX, panY: view.panY };
  };

  const onMouseMove = (e: ReactMouseEvent<HTMLCanvasElement>) => {
    const d = dragRef.current;
    if (!d) return;
    setView((v) => ({
      ...v,
      panX: d.panX + (e.clientX - d.x),
      panY: d.panY + (e.clientY - d.y),
    }));
  };

  const stopDrag = () => {
    dragRef.current = null;
  };

  const zoom = (factor: number) => {
    setView((v) => {
      const newScale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, v.scale * factor));
      const cx = size.w / 2;
      const cy = size.h / 2;
      const ratio = newScale / v.scale;
      return {
        scale: newScale,
        panX: cx - (cx - v.panX) * ratio,
        panY: cy - (cy - v.panY) * ratio,
      };
    });
  };

  const reset = () => {
    setBrightness(1);
    setContrast(1);
    fit();
  };

  const onBrightness = (e: ChangeEvent<HTMLInputElement>) => setBrightness(Number(e.target.value));
  const onContrast = (e: ChangeEvent<HTMLInputElement>) => setContrast(Number(e.target.value));

  return (
    <div className="flex h-full flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2 rounded-md border border-border bg-card/40 p-2 text-xs">
        <Button variant="outline" size="sm" onClick={() => zoom(1.25)} disabled={!image}>
          <ZoomIn className="mr-1 h-3.5 w-3.5" aria-hidden /> +
        </Button>
        <Button variant="outline" size="sm" onClick={() => zoom(1 / 1.25)} disabled={!image}>
          <ZoomOut className="mr-1 h-3.5 w-3.5" aria-hidden /> −
        </Button>
        <Button variant="outline" size="sm" onClick={fit} disabled={!image}>
          <Maximize2 className="mr-1 h-3.5 w-3.5" aria-hidden /> Вместить
        </Button>
        <Button variant="outline" size="sm" onClick={reset} disabled={!image}>
          <RotateCcw className="mr-1 h-3.5 w-3.5" aria-hidden /> Сброс
        </Button>
        <span className="ml-2 tabular-nums text-muted-foreground" data-testid="viewer-zoom">
          {Math.round(view.scale * 100)}%
        </span>
        <label className="ml-auto flex items-center gap-2">
          <span className="text-muted-foreground">Яркость</span>
          <input
            type="range"
            min={0.5}
            max={2}
            step={0.05}
            value={brightness}
            onChange={onBrightness}
            disabled={!image}
            className="w-24 accent-primary"
            aria-label="Яркость"
          />
        </label>
        <label className="flex items-center gap-2">
          <span className="text-muted-foreground">Контраст</span>
          <input
            type="range"
            min={0.5}
            max={2}
            step={0.05}
            value={contrast}
            onChange={onContrast}
            disabled={!image}
            className="w-24 accent-primary"
            aria-label="Контраст"
          />
        </label>
      </div>
      <div
        ref={wrapperRef}
        className="relative aspect-square w-full overflow-hidden rounded-lg border border-border bg-black"
      >
        <canvas
          ref={canvasRef}
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMove}
          onMouseUp={stopDrag}
          onMouseLeave={stopDrag}
          className="block h-full w-full cursor-grab select-none active:cursor-grabbing"
          data-testid="viewer-canvas"
        />
        {(isLoading || error) && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/60 text-sm">
            {isLoading && <span className="text-muted-foreground">Загрузка изображения…</span>}
            {error && <span className="text-destructive">{error}</span>}
          </div>
        )}
      </div>
    </div>
  );
}
