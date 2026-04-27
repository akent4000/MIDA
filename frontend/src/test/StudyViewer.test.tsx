import { render, screen } from "@testing-library/react";
import { describe, it, expect, beforeAll } from "vitest";
import { StudyViewer } from "@/components/StudyViewer";

beforeAll(() => {
  if (!("ResizeObserver" in globalThis)) {
    class RO {
      observe() {}
      unobserve() {}
      disconnect() {}
    }
    (globalThis as unknown as { ResizeObserver: typeof RO }).ResizeObserver = RO;
  }
});

describe("StudyViewer", () => {
  it("shows loading overlay and disables tools when image is missing", () => {
    render(<StudyViewer image={null} isLoading />);
    expect(screen.getByText(/загрузка изображения/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /вместить/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /сброс/i })).toBeDisabled();
    expect(screen.getByLabelText("Яркость")).toBeDisabled();
    expect(screen.getByLabelText("Контраст")).toBeDisabled();
  });

  it("shows error overlay when error is provided", () => {
    render(<StudyViewer image={null} error="Не удалось" />);
    expect(screen.getByText("Не удалось")).toBeInTheDocument();
  });
});
