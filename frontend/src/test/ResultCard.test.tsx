import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { ResultCard } from "@/components/ResultCard";
import type { InferenceResultPublic } from "@/api";

function build(result: Record<string, unknown> | null): InferenceResultPublic {
  return {
    id: "inf-1",
    study_id: "study-1",
    tool_id: "pneumonia_classifier_v1",
    task_id: "task-1",
    status: "done",
    result,
    gradcam_key: null,
    error_message: null,
    created_at: "2026-04-27T00:00:00Z",
    completed_at: "2026-04-27T00:00:01Z",
  };
}

describe("ResultCard", () => {
  it("renders interpretation, probability and confidence band", () => {
    render(
      <ResultCard
        result={build({
          interpretation: "Признаки пневмонии",
          confidence_band: "high",
          raw: { prob: 0.8123, threshold: 0.4396, label: 1, label_name: "pneumonia" },
        })}
      />,
    );
    expect(screen.getByText("Признаки пневмонии")).toBeInTheDocument();
    expect(screen.getByText("81.2%")).toBeInTheDocument();
    expect(screen.getByText("44.0%")).toBeInTheDocument();
    expect(screen.getByText(/высокая/i)).toBeInTheDocument();
  });

  it("falls back gracefully when fields are missing", () => {
    render(<ResultCard result={build(null)} />);
    expect(screen.getAllByText("—").length).toBeGreaterThan(0);
  });
});
