import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, it, expect } from "vitest";
import { Layout } from "@/components/Layout";

describe("Layout", () => {
  it("renders header brand and disclaimer", () => {
    render(
      <MemoryRouter initialEntries={["/"]}>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<div>home</div>} />
          </Route>
        </Routes>
      </MemoryRouter>,
    );
    expect(screen.getByText("MIDA")).toBeInTheDocument();
    expect(screen.getByText(/исследовательский прототип/i)).toBeInTheDocument();
    expect(screen.getByText("home")).toBeInTheDocument();
  });
});
