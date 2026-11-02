import { useEffect, useRef } from 'react';

/**
 * Sets up a DPR-aware canvas that fills its container's width and redraws every
 * animation frame. `aspect` is height / width, and may change between frames.
 */
export function useCanvas(draw: (ctx: CanvasRenderingContext2D, w: number, h: number) => void, aspect: () => number) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drawRef = useRef(draw);
  const aspectRef = useRef(aspect);
  drawRef.current = draw;
  aspectRef.current = aspect;

  useEffect(() => {
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext('2d')!;
    let raf = 0;
    const frame = () => {
      const w = canvas.parentElement!.clientWidth;
      const h = Math.max(1, Math.round(w * aspectRef.current()));
      const dpr = window.devicePixelRatio || 1;
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
        canvas.style.width = `${w}px`;
        canvas.style.height = `${h}px`;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      drawRef.current(ctx, w, h);
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, []);

  return canvasRef;
}
