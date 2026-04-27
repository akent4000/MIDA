import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";

export function NotFoundPage() {
  return (
    <section className="flex flex-col items-center justify-center gap-4 py-16 text-center">
      <h1 className="text-3xl font-semibold">404</h1>
      <p className="text-muted-foreground">Страница не найдена.</p>
      <Button asChild variant="outline">
        <Link to="/">На главную</Link>
      </Button>
    </section>
  );
}
