"""Interactive employee and team event dashboard."""

from fasthtml.common import FastHTML, Div, H1, P, Link, Style, RedirectResponse, serve
from starlette.exceptions import HTTPException
import matplotlib.pyplot as plt
from employee_events import Employee, Team
from utils import load_model, project_root
from base_components import Dropdown, BaseComponent, Radio, MatplotlibViz, DataTable
from combined_components import FormGroup, CombinedComponent


class ReportDropdown(Dropdown):
    def build_component(self, entity_id, model):
        self.label = model.name.title()
        return super().build_component(str(entity_id), model)

    def component_data(self, entity_id, model):
        return model.names()


class Header(BaseComponent):
    def build_component(self, entity_id, model):
        username = model.username(entity_id)
        if not username:
            raise HTTPException(404, "Profile not found")
        return Div(H1(f"{model.name.title()} Performance"), P(username[0][0]))


class LineChart(MatplotlibViz):
    def visualization(self, asset_id, model):
        data = model.event_counts(asset_id).fillna(0).set_index("event_date").sort_index().cumsum()
        data.columns = ["Positive", "Negative"]
        _, ax = plt.subplots(figsize=(7, 4))
        if not data.empty:
            data.plot(ax=ax, color=["#16705b", "#c54b4b"])
        self.set_axis_styling(ax, bordercolor="black", fontcolor="black")
        ax.set(title="Cumulative Event Counts", xlabel="Date", ylabel="Events")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()


class BarChart(MatplotlibViz):
    predictor = load_model()

    def visualization(self, asset_id, model):
        data = model.model_data(asset_id).fillna(0)
        prediction = self.predictor.predict_proba(data)[:, 1]
        pred = float(prediction.mean() if model.name == "team" else prediction[0])
        _, ax = plt.subplots(figsize=(7, 4))
        ax.barh([""], [pred], color=plt.cm.RdYlGn_r(pred))
        ax.set_xlim(0, 1)
        ax.set_title("Predicted Recruitment Risk", fontsize=20)
        ax.set_xlabel("Probability (team view: mean employee probability)")
        ax.text(min(pred + 0.02, 0.85), 0, f"{pred:.1%}", va="center")
        self.set_axis_styling(ax, bordercolor="black", fontcolor="black")
        plt.tight_layout()


class Visualizations(CombinedComponent):
    children = [LineChart(), BarChart()]
    outer_div_type = Div(cls="grid")


class NotesTable(DataTable):
    def component_data(self, entity_id, model):
        return model.notes(entity_id)


class DashboardFilters(FormGroup):
    id = "top-filters"
    action = "/update_data"
    method = "POST"
    children = [
        Radio(
            values=["Employee", "Team"],
            name="profile_type",
            hx_get="/update_dropdown",
            hx_target="#selector",
        ),
        ReportDropdown(id="selector", name="user-selection"),
    ]


class Report(CombinedComponent):
    children = [Header(), DashboardFilters(), Visualizations(), NotesTable()]


app = FastHTML(
    hdrs=(
        Link(
            rel="stylesheet", href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css"
        ),
        Style((project_root / "assets" / "report.css").read_text()),
    )
)
report = Report()


@app.get("/")
def index():
    return report(1, Employee())


@app.get("/employee/{id}")
def employee(id: int):
    return report(id, Employee())


@app.get("/team/{id}")
def team(id: int):
    return report(id, Team())


@app.get("/update_dropdown")
def update_dropdown(profile_type: str):
    if profile_type not in ("Employee", "Team"):
        raise HTTPException(400, "Invalid profile type")
    return DashboardFilters.children[1](None, Team() if profile_type == "Team" else Employee())


@app.post("/update_data")
async def update_data(r):
    data = await r.form()
    profile_type = data.get("profile_type")
    if profile_type not in ("Employee", "Team"):
        raise HTTPException(400, "Invalid profile type")
    try:
        entity_id = int(data["user-selection"])
    except (KeyError, ValueError):
        raise HTTPException(400, "A numeric profile ID is required")
    return RedirectResponse(f"/{profile_type.lower()}/{entity_id}", status_code=303)


if __name__ == "__main__":
    serve()
