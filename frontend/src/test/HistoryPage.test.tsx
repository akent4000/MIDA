import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { HistoryPage } from "@/pages/HistoryPage";

const listMock = vi.fn();

vi.mock("@/api", async () => {
  const actual = await vi.importActual<typeof import("@/api")>("@/api");
  return {
    ...actual,
    StudiesService: {
      listStudiesApiV1StudiesGet: (...args: unknown[]) => listMock(...args),
      uploadStudyApiV1StudiesPost: vi.fn(),
      getStudyApiV1StudiesStudyIdGet: vi.fn(),
      getStudyImageApiV1StudiesStudyIdImageGet: vi.fn(),
    },
  };
});

function renderHistory() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={["/history"]}>
        <Routes>
          <Route path="/history" element={<HistoryPage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("HistoryPage", () => {
  beforeEach(() => {
    listMock.mockReset();
  });

  it("renders empty-state hint when no studies", async () => {
    listMock.mockResolvedValueOnce([]);
    renderHistory();
    await screen.findByText(/история пуста/i);
  });

  it("renders studies in a table with open links", async () => {
    listMock.mockResolvedValueOnce([
      {
        id: "11111111-2222-3333-4444-555555555555",
        file_format: "dicom",
        file_size: 2048,
        anonymized: true,
        file_key: "k",
        created_at: "2026-04-27T10:00:00Z",
        dicom_metadata: {},
      },
      {
        id: "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        file_format: "png",
        file_size: 1024 * 1024 * 3,
        anonymized: false,
        file_key: "k2",
        created_at: "2026-04-26T12:00:00Z",
        dicom_metadata: {},
      },
    ]);
    renderHistory();
    const links = await screen.findAllByRole("link", { name: /открыть/i });
    expect(links).toHaveLength(2);
    expect(links[0]).toHaveAttribute("href", "/studies/11111111-2222-3333-4444-555555555555");
    expect(screen.getByText("dicom")).toBeInTheDocument();
    expect(screen.getByText("png")).toBeInTheDocument();
    expect(screen.getByText("3.0 MB")).toBeInTheDocument();
  });

  it("shows error alert when the request fails", async () => {
    listMock.mockRejectedValueOnce(new Error("network down"));
    renderHistory();
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("network down");
  });
});
