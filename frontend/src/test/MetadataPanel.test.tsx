import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { MetadataPanel } from "@/components/MetadataPanel";
import type { StudyPublic } from "@/api";

const study: StudyPublic = {
  id: "abc",
  file_format: "dicom",
  file_size: 2048,
  anonymized: true,
  file_key: "studies/abc/raw.dcm",
  created_at: "2026-04-27T10:00:00Z",
  dicom_metadata: {
    PatientSex: "F",
    Modality: "CR",
    Rows: 1024,
  },
};

describe("MetadataPanel", () => {
  it("renders file info and DICOM metadata rows", () => {
    render(<MetadataPanel study={study} />);
    expect(screen.getByText("Файл")).toBeInTheDocument();
    expect(screen.getByText("DICOM")).toBeInTheDocument();
    expect(screen.getByText("2.0 KB")).toBeInTheDocument();
    expect(screen.getByText("да")).toBeInTheDocument();
    expect(screen.getByText("DICOM-метаданные")).toBeInTheDocument();
    expect(screen.getByText("PatientSex")).toBeInTheDocument();
    expect(screen.getByText("F")).toBeInTheDocument();
    expect(screen.getByText("Modality")).toBeInTheDocument();
    expect(screen.getByText("1024")).toBeInTheDocument();
  });

  it("shows fallback when DICOM metadata is empty", () => {
    render(<MetadataPanel study={{ ...study, dicom_metadata: {} }} />);
    expect(screen.getByText(/метаданные dicom отсутствуют/i)).toBeInTheDocument();
  });
});
