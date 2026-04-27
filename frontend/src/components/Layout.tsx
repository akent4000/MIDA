import { NavLink, Outlet } from "react-router-dom";
import { cn } from "@/lib/utils";

const navItems = [
  { to: "/", label: "Загрузка", end: true },
  { to: "/history", label: "История" },
  { to: "/settings", label: "Настройки" },
];

export function Layout() {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="border-b border-border bg-card/40 backdrop-blur supports-[backdrop-filter]:bg-card/30">
        <div className="container flex h-14 items-center justify-between">
          <NavLink to="/" className="flex items-center gap-2 font-semibold">
            <span className="inline-block h-6 w-6 rounded bg-primary" aria-hidden />
            <span>MIDA</span>
            <span className="text-xs font-normal text-muted-foreground">
              Medical Imaging Recognition Assistant
            </span>
          </NavLink>
          <nav className="flex items-center gap-1">
            {navItems.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  cn(
                    "rounded-md px-3 py-1.5 text-sm transition-colors",
                    isActive
                      ? "bg-secondary text-secondary-foreground"
                      : "text-muted-foreground hover:bg-secondary/60 hover:text-foreground",
                  )
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>

      <main className="container flex-1 py-6">
        <Outlet />
      </main>

      <footer className="border-t border-border py-4">
        <div className="container text-xs text-muted-foreground">
          <strong className="text-foreground">Внимание:</strong> исследовательский прототип, не
          медицинское изделие. Результаты не предназначены для клинических решений.
        </div>
      </footer>
    </div>
  );
}
