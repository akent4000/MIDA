import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { UploadPage } from "@/pages/UploadPage";

const uploadMock = vi.fn();

vi.mock("@/api", async () => {
  const actual = await vi.importActual<typeof import("@/api")>("@/api");
  return {
    ...actual,
    StudiesService: {
      uploadStudyApiV1StudiesPost: (...args: unknown[]) => uploadMock(...args),
      getStudyApiV1StudiesStudyIdGet: vi.fn(),
      getStudyImageApiV1StudiesStudyIdImageGet: vi.fn(),
    },
  };
});

function renderWithProviders() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={["/"]}>
        <Routes>
          <Route path="/" element={<UploadPage />} />
          <Route path="/studies/:id" element={<div>study-page</div>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("UploadPage", () => {
  beforeEach(() => {
    uploadMock.mockReset();
  });

  it("uploads a selected file and navigates to study page", async () => {
    uploadMock.mockResolvedValueOnce({
      id: "study-1",
      file_format: "dicom",
      file_size: 1,
      anonymized: false,
      file_key: "k",
      created_at: "2026-04-27T00:00:00Z",
      dicom_metadata: {},
    });
    const user = userEvent.setup();
    renderWithProviders();

    const input = screen.getByTestId("upload-input") as HTMLInputElement;
    const file = new File(["dummy"], "scan.dcm", { type: "application/dicom" });
    await user.upload(input, file);

    expect(screen.getByText("scan.dcm")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /загрузить/i }));

    await waitFor(() => {
      expect(uploadMock).toHaveBeenCalledTimes(1);
    });
    expect(uploadMock.mock.calls[0][0]).toMatchObject({ formData: { file } });
    await screen.findByText("study-page");
  });

  it("shows an error message when upload fails", async () => {
    uploadMock.mockRejectedValueOnce(new Error("boom"));
    const user = userEvent.setup();
    renderWithProviders();

    const input = screen.getByTestId("upload-input") as HTMLInputElement;
    await user.upload(input, new File(["x"], "x.dcm"));
    await user.click(screen.getByRole("button", { name: /загрузить/i }));

    await screen.findByRole("alert");
    expect(screen.getByRole("alert")).toHaveTextContent("boom");
  });
});
